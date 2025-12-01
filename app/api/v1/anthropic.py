"""Anthropic API 兼容路由"""

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from typing import Optional, Dict, Any
from fastapi.responses import StreamingResponse

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient
from app.models.anthropic_schema import (
    AnthropicChatRequest,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse
)
from app.services.anthropic.converter import AnthropicConverter


router = APIRouter(prefix="/messages", tags=["Anthropic"])


def log_anthropic_headers(request: Request):
    """记录 Anthropic 相关请求头（用于调试）"""
    anthropic_version = request.headers.get("anthropic-version")
    anthropic_beta = request.headers.get("anthropic-beta")
    if anthropic_version or anthropic_beta:
        logger.debug(
            f"[Anthropic] 请求头 - version: {anthropic_version}, beta: {anthropic_beta}"
        )


def _estimate_tokens(text: str) -> int:
    """估算文本的 token 数量（简单估算：1 token ≈ 4 字符）"""
    if not text:
        return 0
    # 更准确的估算：考虑中文字符（通常 1 中文字符 ≈ 1.5 tokens）
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


def _count_message_tokens(messages: list, system: Optional[str] = None) -> int:
    """计算消息的 token 数量"""
    total = 0
    
    # 系统提示词
    if system:
        if isinstance(system, str):
            total += _estimate_tokens(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += _estimate_tokens(block.get("text", ""))
    
    # 消息内容
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        total += _estimate_tokens(block.get("text", ""))
                    elif block_type == "image":
                        # 图片通常占用较多 token，估算为 85 tokens（基于 Claude 文档）
                        total += 85
                    elif block_type == "document":
                        # 文档估算为 100 tokens
                        total += 100
    
    return total


@router.post("/count_tokens", response_model=AnthropicCountTokensResponse)
async def count_tokens(
    request: Request,
    count_request: AnthropicCountTokensRequest,
    _: Optional[str] = Depends(auth_manager.verify)
) -> AnthropicCountTokensResponse:
    """计算消息的 token 数量（Anthropic 兼容接口）"""
    try:
        log_anthropic_headers(request)
        logger.debug("[Anthropic] 收到 token 计数请求")
        
        # 转换消息为字典格式以便计算
        messages_dict = [msg.model_dump() for msg in count_request.messages]
        
        # 计算 token 数量
        input_tokens = _count_message_tokens(messages_dict, count_request.system)
        
        logger.debug(f"[Anthropic] Token 计数: {input_tokens} tokens (模型: {count_request.model})")
        
        return AnthropicCountTokensResponse(input_tokens=input_tokens)
        
    except Exception as e:
        logger.error(f"[Anthropic] Token 计数失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": f"Token 计数失败: {str(e)}"
                }
            }
        )


@router.post("", response_model=None)
async def create_message(
    http_request: Request,
    request: AnthropicChatRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """创建消息（Anthropic 兼容接口）"""
    try:
        log_anthropic_headers(http_request)
        logger.info("[Anthropic] 收到 Anthropic 格式的聊天请求")
        
        # 转换 Anthropic 请求为 OpenAI 格式
        openai_request = AnthropicConverter.to_openai_format(request.model_dump())
        
        # 调用 Grok 客户端
        result = await GrokClient.openai_to_grok(openai_request)
        
        # 流式响应
        if request.stream:
            # 转换流式响应为 Anthropic 格式
            async def anthropic_stream():
                async for chunk in AnthropicConverter.to_anthropic_stream(result, request.model):
                    yield chunk
            
            return StreamingResponse(
                content=anthropic_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # 非流式响应 - 将 Pydantic 对象转换为字典
        # result 是 OpenAIChatCompletionResponse 对象，需要转换为字典
        if result is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Grok API 返回空响应，请检查是否配置了有效的 Grok token"
                    }
                }
            )
        
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            logger.error(f"[Anthropic] 未知的响应类型: {type(result)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "type": "error",
                    "error": {
                        "type": "internal_error",
                        "message": f"未知的响应类型: {type(result)}"
                    }
                }
            )
        
        # 转换为 Anthropic 格式
        anthropic_response = AnthropicConverter.to_anthropic_response(result_dict, request.model)
        return anthropic_response
        
    except GrokApiException as e:
        logger.error(f"[Anthropic] Grok API错误: {e} - 详情: {e.details}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": e.error_code or "api_error",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"[Anthropic] 处理失败: {e}\n{error_detail}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": f"服务器内部错误: {str(e)}"
                }
            }
        )







