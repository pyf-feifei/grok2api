"""聊天API路由 - OpenAI兼容的聊天接口（兼容 Anthropic 格式）"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Union, List, Dict, Any
from fastapi.responses import StreamingResponse

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient
from app.services.anthropic.converter import AnthropicConverter
from app.models.openai_schema import OpenAIChatRequest


router = APIRouter(prefix="/chat", tags=["聊天"])


@router.post("/completions", response_model=None)
async def chat_completions(request: OpenAIChatRequest, _: Optional[str] = Depends(auth_manager.verify)):
    """创建聊天补全（支持流式和非流式）"""
    try:
        logger.info("[Chat] 收到聊天请求")
        
        # 记录原始请求详情
        request_dict = request.model_dump()
        logger.info(f"[Chat] 原始请求详情:")
        logger.info(f"[Chat]   - model: {request_dict.get('model')}")
        logger.info(f"[Chat]   - system: {request_dict.get('system')} (类型: {type(request_dict.get('system'))})")
        logger.info(f"[Chat]   - messages 数量: {len(request_dict.get('messages', []))}")
        for idx, msg in enumerate(request_dict.get('messages', [])):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
            logger.info(f"[Chat]   - messages[{idx}]: role={role}, content类型={type(content)}, content预览={content_preview}")
        
        # 兼容处理：如果请求中有独立的 system 参数，但没有在 messages 中，则添加到 messages 开头
        system_param = request_dict.get('system')
        if system_param:
            # 检查 messages 中是否已有 system 角色
            has_system_in_messages = any(
                msg.get('role') == 'system' 
                for msg in request_dict.get('messages', [])
            )
            
            if not has_system_in_messages:
                logger.info("[Chat] 检测到独立的 system 参数，但 messages 中没有 system 消息，正在添加...")
                # 提取 system 内容（支持字符串和数组格式）
                system_content = AnthropicConverter._extract_system_content(system_param)
                if system_content:
                    # 在 messages 开头插入 system 消息
                    request_dict['messages'].insert(0, {
                        "role": "system",
                        "content": system_content
                    })
                    logger.info(f"[Chat] 已添加 system 消息到 messages 开头: {system_content[:200] + '...' if len(system_content) > 200 else system_content}")
                else:
                    logger.warning("[Chat] system 参数存在但提取的内容为空")
        
        # 调用Grok客户端
        result = await GrokClient.openai_to_grok(request_dict)
        
        # 流式响应
        if request.stream:
            return StreamingResponse(
                content=result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # 非流式响应
        return result
        
    except GrokApiException as e:
        logger.error(f"[Chat] Grok API错误: {e} - 详情: {e.details}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": e.error_code or "grok_api_error",
                    "code": e.error_code or "unknown"
                }
            }
        )
    except Exception as e:
        logger.error(f"[Chat] 处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "服务器内部错误",
                    "type": "internal_error",
                    "code": "internal_server_error"
                }
            }
        )
