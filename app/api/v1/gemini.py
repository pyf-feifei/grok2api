"""Gemini API路由 - Google Gemini兼容的图生视频接口"""

import re
from fractions import Fraction
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any, List
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient


router = APIRouter(tags=["Gemini"])


class InlineData(BaseModel):
    """内联数据"""
    mimeType: str = Field(..., description="MIME类型")
    data: str = Field(..., description="Base64编码的数据")


class Part(BaseModel):
    """内容部分"""
    text: Optional[str] = Field(None, description="文本内容")
    inlineData: Optional[InlineData] = Field(None, description="内联数据")


class Content(BaseModel):
    """内容"""
    parts: List[Part] = Field(..., description="内容部分列表")


class GenerationConfig(BaseModel):
    """生成配置"""
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="采样温度")
    topK: Optional[int] = Field(40, ge=1, description="Top-K采样")
    topP: Optional[float] = Field(0.95, ge=0, le=1, description="Top-P采样")


class GeminiGenerateContentRequest(BaseModel):
    """Gemini生成内容请求"""
    contents: List[Content] = Field(..., description="内容列表")
    generationConfig: Optional[GenerationConfig] = Field(None, description="生成配置")


@router.post("/models/{model_name}/generateContent", response_model=None)
async def gemini_generate_content(
    model_name: str,
    request: GeminiGenerateContentRequest,
    key: Optional[str] = Query(None, description="API密钥（兼容Gemini格式）"),
    _: Optional[str] = Depends(auth_manager.verify)
):
    """Gemini兼容的图生视频接口
    
    将Gemini格式的请求转换为Grok视频生成请求
    """
    try:
        logger.info("[Gemini] 收到图生视频请求")
        
        # 提取内容和图片
        if not request.contents or not request.contents[0].parts:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "内容不能为空", "code": "invalid_argument"}}
            )
        
        parts = request.contents[0].parts
        text_parts = [p.text for p in parts if p.text]
        image_parts = [p.inlineData for p in parts if p.inlineData]
        
        if not image_parts:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "需要至少一张图片", "code": "invalid_argument"}}
            )
        
        # 提取提示词（从文本中提取，支持分辨率信息）
        prompt = " ".join(text_parts) if text_parts else "生成视频"
        
        # 解析分辨率（从提示词中提取，如 "Video resolution: 1280x720"）
        width = 1280
        height = 720
        aspect_ratio = None
        video_length = None
        
        resolution_match = re.search(r'Video resolution:\s*(\d+)x(\d+)', prompt, re.IGNORECASE)
        if resolution_match:
            width = int(resolution_match.group(1))
            height = int(resolution_match.group(2))
            # 计算宽高比
            ratio = Fraction(width, height)
            aspect_ratio = f"{ratio.numerator}:{ratio.denominator}"
        
        # 尝试从提示词中提取时长（如 "Video length: 6 seconds"）
        length_match = re.search(r'Video length:\s*(\d+)', prompt, re.IGNORECASE)
        if length_match:
            video_length = int(length_match.group(1))
        
        # 清理提示词（移除分辨率信息）
        prompt = re.sub(r'Video resolution:\s*\d+x\d+\.?', '', prompt, flags=re.IGNORECASE).strip()
        prompt = re.sub(r'Video length:\s*\d+\s*seconds?\.?', '', prompt, flags=re.IGNORECASE).strip()
        
        # 使用第一张图片
        image_data = image_parts[0]
        image_base64 = image_data.data
        
        # 构建OpenAI格式的请求
        openai_request = {
            "model": "grok-imagine-0.9",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_data.mimeType};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "stream": False,
            "aspect_ratio": aspect_ratio,
            "duration": video_length  # 使用 duration 符合 OpenAI Sora API 格式
        }
        
        # 调用Grok客户端
        result = await GrokClient.openai_to_grok(openai_request)
        
        # 转换为Gemini格式的响应
        logger.info(f"[Gemini] 收到响应类型: {type(result)}")
        
        # 将Pydantic模型转换为字典（如果返回的是模型对象）
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            logger.warning(f"[Gemini] 未预期的响应类型: {type(result)}")
            result_dict = {"choices": []}
        
        logger.info(f"[Gemini] 转换后的字典，keys: {list(result_dict.keys())}")
        
        # 提取视频内容
        video_content = ""
        if "choices" in result_dict and result_dict["choices"]:
            if result_dict["choices"][0].get("message", {}).get("content"):
                video_content = result_dict["choices"][0]["message"]["content"]
                logger.info(f"[Gemini] 提取到视频内容长度: {len(video_content)}")
        
        # 构建Gemini格式响应
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": video_content
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": result_dict.get("usage", {}) if isinstance(result_dict.get("usage"), dict) else {}
        }
        
        logger.info(f"[Gemini] 转换后的Gemini响应，candidates数量: {len(gemini_response['candidates'])}")
        return gemini_response
        
    except HTTPException:
        raise
    except GrokApiException as e:
        logger.error(f"[Gemini] Grok API错误: {e} - 详情: {e.details}")
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
        logger.error(f"[Gemini] 处理失败: {e}")
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

