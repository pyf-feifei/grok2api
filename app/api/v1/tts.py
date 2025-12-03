"""TTS API路由 - OpenAI兼容的文本转语音接口"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from typing import Optional

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.tts import GrokTTSService
from app.models.openai_schema import OpenAITTSRequest


router = APIRouter(prefix="/audio", tags=["语音"])


@router.post("/speech")
async def create_speech(
    request: OpenAITTSRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """
    创建语音（文本转语音）
    
    标准OpenAI兼容接口，支持将文本转换为语音。
    注意：Grok返回WAV格式音频，voice和speed参数可能不被支持。
    """
    try:
        logger.info(f"[TTS] 收到TTS请求，文本长度: {len(request.input)} 字符")
        logger.debug(f"[TTS] 请求参数 - model: {request.model}, voice: {request.voice}, format: {request.response_format}")
        
        # 验证文本长度
        if len(request.input) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "输入文本不能为空",
                        "type": "invalid_request_error",
                        "code": "empty_input"
                    }
                }
            )
        
        # 限制文本长度（避免过长）
        max_length = 5000  # 可以根据需要调整
        if len(request.input) > max_length:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"输入文本过长，最大支持 {max_length} 字符",
                        "type": "invalid_request_error",
                        "code": "text_too_long"
                    }
                }
            )
        
        # 调用TTS服务
        try:
            audio_data = await GrokTTSService.text_to_speech(
                request.input,
                model=request.model if request.model != "tts-1" else "grok-4.1"
            )
        except GrokApiException as e:
            logger.error(f"[TTS] Grok API错误: {e} - 详情: {e.details}")
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
        
        # 确定Content-Type
        # Grok返回WAV格式，但根据request.response_format设置Content-Type
        content_type_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac"
        }
        
        response_format = request.response_format or "wav"
        content_type = content_type_map.get(response_format.lower(), "audio/wav")
        
        # 注意：如果请求的格式不是wav，但Grok只返回wav，这里仍然返回wav
        # 如果需要格式转换，可以后续添加转换逻辑
        if response_format.lower() != "wav":
            logger.warning(f"[TTS] 请求格式 {response_format} 可能不被支持，返回WAV格式")
            content_type = "audio/wav"
        
        logger.info(f"[TTS] 成功生成语音，大小: {len(audio_data)} bytes, Content-Type: {content_type}")
        
        # 返回音频流
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format.lower()}"',
                "Content-Length": str(len(audio_data))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TTS] 处理失败: {e}")
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





