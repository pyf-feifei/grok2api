"""Anthropic API 兼容路由"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient
from app.models.anthropic_schema import AnthropicChatRequest
from app.services.anthropic.converter import AnthropicConverter


router = APIRouter(prefix="/messages", tags=["Anthropic"])


@router.post("", response_model=None)
async def create_message(request: AnthropicChatRequest, _: Optional[str] = Depends(auth_manager.verify)):
    """创建消息（Anthropic 兼容接口）"""
    try:
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
        
        # 非流式响应 - 转换为 Anthropic 格式
        anthropic_response = AnthropicConverter.to_anthropic_response(result, request.model)
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
        logger.error(f"[Anthropic] 处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": "服务器内部错误"
                }
            }
        )







