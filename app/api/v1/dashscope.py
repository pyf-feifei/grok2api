"""DashScope兼容API路由 - 通义万相API兼容接口"""

import base64
import re
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.core.config import setting
from app.services.grok.client import GrokClient
from app.services.grok.task_manager import task_manager, TaskStatus
from app.models.grok_models import Models


router = APIRouter(prefix="/v1/services/aigc", tags=["DashScope兼容"])

# Bearer安全方案（用于DashScope兼容接口）
dashscope_security = HTTPBearer(auto_error=False)


# ==================== 请求模型 ====================

class Text2ImageRequest(BaseModel):
    """文生图请求"""
    model: str = Field(default="wan2.5-t2i-preview", description="模型名称")
    input: Dict[str, Any] = Field(..., description="输入参数")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="额外参数")


class Image2ImageRequest(BaseModel):
    """图生图请求"""
    model: str = Field(default="qwen-image-edit-plus", description="模型名称")
    input: Dict[str, Any] = Field(..., description="输入参数")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="额外参数")


class Text2VideoRequest(BaseModel):
    """文生视频请求"""
    model: str = Field(default="wan2.5-i2v-preview", description="模型名称")
    input: Dict[str, Any] = Field(..., description="输入参数")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="额外参数")


class Image2VideoRequest(BaseModel):
    """图生视频请求"""
    model: str = Field(default="wan2.5-i2v-preview", description="模型名称")
    input: Dict[str, Any] = Field(..., description="输入参数")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="额外参数")


class TaskQueryRequest(BaseModel):
    """任务查询请求"""
    task_id: str = Field(..., description="任务ID")


# ==================== 辅助函数 ====================

def _extract_base64_image(data_url: str) -> Optional[str]:
    """从Base64 Data URL中提取图片数据"""
    if not data_url.startswith("data:image/"):
        return None

    # 提取base64部分
    parts = data_url.split(",", 1)
    if len(parts) != 2:
        return None

    return parts[1]


def _parse_size(size_str: str) -> Optional[Dict[str, int]]:
    """解析尺寸字符串（如 "1280*720"）"""
    if not size_str:
        return None

    try:
        parts = size_str.split("*")
        if len(parts) != 2:
            return None
        return {"width": int(parts[0]), "height": int(parts[1])}
    except ValueError:
        return None


def _convert_to_grok_model(dashscope_model: str, task_type: str) -> str:
    """将DashScope模型名转换为Grok模型名

    Args:
        dashscope_model: DashScope模型名
        task_type: 任务类型（text2image, image2image, text2video, image2video）
    """
    # 根据任务类型返回对应的Grok模型
    if task_type in ["text2video", "image2video"]:
        return "grok-imagine-0.9"  # 视频生成模型
    else:
        # 图片生成使用grok-4.1或grok-imagine-0.9
        return "grok-imagine-0.9"  # 使用imagine模型进行图片生成


def _build_grok_messages(prompt: str, images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """构建Grok消息格式"""
    messages = []
    content = []

    # 添加图片
    if images:
        for img in images:
            if img.startswith("data:image/"):
                # Base64 Data URL
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            elif img.startswith("http"):
                # HTTP URL
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })

    # 添加文本提示词
    if prompt:
        content.append({
            "type": "text",
            "text": prompt
        })

    if content:
        messages.append({
            "role": "user",
            "content": content
        })

    return messages


async def _process_async_task(task_id: str, grok_model: str, messages: List[Dict[str, Any]],
                              task_type: str, parameters: Optional[Dict[str, Any]] = None,
                              auth_token: Optional[str] = None):
    """处理异步任务"""
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING)

        # 构建Grok请求
        request_data = {
            "model": grok_model,
            "messages": messages,
            "stream": False
        }

        # 添加视频参数
        if task_type in ["text2video", "image2video"] and parameters:
            if "resolution" in parameters:
                # 将分辨率转换为aspect_ratio
                resolution = parameters["resolution"]
                if resolution == "480P":
                    request_data["aspect_ratio"] = "16:9"
                elif resolution == "720P":
                    request_data["aspect_ratio"] = "16:9"
                elif resolution == "1080P":
                    request_data["aspect_ratio"] = "16:9"

        # 调用Grok客户端（这会自动缓存图片和视频）
        result = await GrokClient.openai_to_grok(request_data)

        # 提取结果
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            content = choice.message.content if hasattr(
                choice.message, "content") else ""

            # 提取图片或视频URL
            if task_type in ["text2image", "image2image"]:
                # 从markdown中提取图片URL
                image_urls = re.findall(r'!\[.*?\]\((.*?)\)', content)
                # 也尝试从HTML img标签中提取
                if not image_urls:
                    image_urls = re.findall(
                        r'<img[^>]+src=["\'](.*?)["\']', content)

                if image_urls:
                    # 处理URL：识别本地缓存路径和原始路径
                    processed_urls = []
                    for url in image_urls:
                        # 跳过本地完整URL（http://localhost:8001/images/...）
                        if url.startswith("http://localhost") or url.startswith("https://localhost"):
                            # 这是本地服务URL，提取路径部分
                            if "/images/" in url:
                                filename = url.split("/images/")[1]
                                # 转换为原始路径格式
                                original_path = f"/{filename.replace('-', '/')}"
                                processed_urls.append(original_path)
                            continue
                        elif url.startswith("http"):
                            # 完整URL，提取相对路径
                            if "assets.grok.com" in url:
                                # 提取路径部分
                                path = url.split("assets.grok.com")[1]
                                processed_urls.append(path)
                            else:
                                # 其他URL，跳过（不是Grok assets）
                                continue
                        elif url.startswith("/images/"):
                            # 本地缓存路径，需要从文件名还原原始路径
                            # 格式：/images/users-xxx-generated-xxx-image.jpg
                            # 需要转换为：/users/xxx/generated/xxx/image.jpg
                            filename = url.replace(
                                "/images/", "").replace("-", "/")
                            processed_urls.append(f"/{filename}")
                        elif url.startswith("/"):
                            # 相对路径（Grok assets路径），直接使用
                            processed_urls.append(url)

                    if processed_urls:
                        # 如果有auth_token，尝试下载并缓存
                        if auth_token:
                            from app.services.grok.cache import image_cache_service
                            from app.core.config import setting

                            cached_urls = []
                            for url_path in processed_urls:
                                try:
                                    # 下载并缓存图片
                                    cache_path = await image_cache_service.download_image(url_path, auth_token)
                                    if cache_path:
                                        # 转换为本地访问URL
                                        img_path = url_path.replace(
                                            '/', '-').lstrip('-')
                                        base_url = setting.global_config.get(
                                            "base_url", "")
                                        local_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        cached_urls.append(local_url)
                                    else:
                                        # 缓存失败，使用原始URL
                                        if url_path.startswith("/"):
                                            cached_urls.append(
                                                f"https://assets.grok.com{url_path}")
                                        else:
                                            cached_urls.append(url_path)
                                except Exception as e:
                                    logger.warning(
                                        f"[DashScope] 缓存图片失败: {url_path} - {e}")
                                    if url_path.startswith("/"):
                                        cached_urls.append(
                                            f"https://assets.grok.com{url_path}")
                                    else:
                                        cached_urls.append(url_path)

                            task_manager.update_task(task_id, status=TaskStatus.SUCCESS,
                                                     result={"image_urls": cached_urls})
                        else:
                            # 没有auth_token，直接使用原始URL
                            final_urls = [f"https://assets.grok.com{url}" if url.startswith("/") else url
                                          for url in processed_urls]
                            task_manager.update_task(task_id, status=TaskStatus.SUCCESS,
                                                     result={"image_urls": final_urls})
                    else:
                        task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                                 error="未找到有效的图片URL")
                else:
                    task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                             error="未找到生成的图片")
            elif task_type in ["text2video", "image2video"]:
                # 从HTML中提取视频URL
                video_urls = re.findall(
                    r'<video[^>]+src=["\'](.*?)["\']', content)
                # 也尝试从markdown链接中提取
                if not video_urls:
                    video_urls = re.findall(
                        r'!\[.*?\]\((.*?\.(mp4|webm|mov))\)', content)

                if video_urls:
                    # 提取URL（如果是元组，取第一个元素）
                    video_urls = [url[0] if isinstance(
                        url, tuple) else url for url in video_urls]

                    # 处理URL
                    processed_urls = []
                    for url in video_urls:
                        # 跳过本地完整URL（http://localhost:8001/images/...）
                        if url.startswith("http://localhost") or url.startswith("https://localhost"):
                            # 这是本地服务URL，提取路径部分
                            if "/images/" in url:
                                filename = url.split("/images/")[1]
                                # 转换为原始路径格式
                                original_path = f"/{filename.replace('-', '/')}"
                                processed_urls.append(original_path)
                            continue
                        elif url.startswith("http"):
                            if "assets.grok.com" in url:
                                path = url.split("assets.grok.com")[1]
                                processed_urls.append(path)
                            else:
                                # 其他URL，跳过（不是Grok assets）
                                continue
                        elif url.startswith("/images/"):
                            filename = url.replace(
                                "/images/", "").replace("-", "/")
                            processed_urls.append(f"/{filename}")
                        elif url.startswith("/"):
                            # 相对路径（Grok assets路径），直接使用
                            processed_urls.append(url)

                    if processed_urls:
                        # 如果有auth_token，尝试下载并缓存
                        if auth_token:
                            from app.services.grok.cache import video_cache_service
                            from app.core.config import setting

                            cached_urls = []
                            for url_path in processed_urls:
                                try:
                                    # 下载并缓存视频
                                    cache_path = await video_cache_service.download_video(url_path, auth_token)
                                    if cache_path:
                                        # 转换为本地访问URL
                                        video_path = url_path.replace(
                                            '/', '-').lstrip('-')
                                        base_url = setting.global_config.get(
                                            "base_url", "")
                                        local_url = f"{base_url}/images/{video_path}" if base_url else f"/images/{video_path}"
                                        cached_urls.append(local_url)
                                    else:
                                        # 缓存失败，使用原始URL
                                        if url_path.startswith("/"):
                                            cached_urls.append(
                                                f"https://assets.grok.com{url_path}")
                                        else:
                                            cached_urls.append(url_path)
                                except Exception as e:
                                    logger.warning(
                                        f"[DashScope] 缓存视频失败: {url_path} - {e}")
                                    if url_path.startswith("/"):
                                        cached_urls.append(
                                            f"https://assets.grok.com{url_path}")
                                    else:
                                        cached_urls.append(url_path)

                            task_manager.update_task(task_id, status=TaskStatus.SUCCESS,
                                                     result={"video_urls": cached_urls})
                        else:
                            # 没有auth_token，直接使用原始URL
                            final_urls = [f"https://assets.grok.com{url}" if url.startswith("/") else url
                                          for url in processed_urls]
                            task_manager.update_task(task_id, status=TaskStatus.SUCCESS,
                                                     result={"video_urls": final_urls})
                    else:
                        task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                                 error="未找到有效的视频URL")
                else:
                    task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                             error="未找到生成的视频")
            else:
                task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                         error="未知任务类型")
        else:
            task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                     error="响应格式错误")

    except Exception as e:
        logger.error(f"[DashScope] 任务处理失败: {task_id} - {e}")
        task_manager.update_task(task_id, status=TaskStatus.FAILED,
                                 error=str(e))


# ==================== API端点 ====================

@router.post("/text2image/image-synthesis")
async def text2image(
    request: Text2ImageRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        dashscope_security),
    async_header: Optional[str] = Header(None, alias="X-DashScope-Async")
):
    """文生图接口（异步模式）"""
    try:
        # 验证认证
        _ = auth_manager.verify(credentials)

        # 检查是否为异步模式
        is_async = async_header == "enable"

        # 提取参数
        prompt = request.input.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="缺少prompt参数")

        parameters = request.parameters or {}
        # size是必填参数
        size = parameters.get("size")
        if not size:
            raise HTTPException(status_code=400, detail="缺少size参数（必填）")
        n = parameters.get("n", 1)

        # 转换为Grok模型
        grok_model = _convert_to_grok_model(request.model, "text2image")

        # 构建消息
        messages = _build_grok_messages(prompt)

        if is_async:
            # 异步模式：创建任务
            task_id = task_manager.create_task("text2image", {
                "prompt": prompt,
                "size": size,
                "n": n
            })

            # 获取auth_token（从credentials中提取）
            auth_token = None
            if credentials and credentials.credentials:
                # 从Bearer token中提取，需要转换为Grok token格式
                # 这里需要从token_manager获取实际的Grok token
                from app.services.grok.token import token_manager
                from app.models.grok_models import Models
                try:
                    # 获取一个可用的token
                    token = token_manager.get_token(grok_model)
                    if token:
                        auth_token = token
                except:
                    pass

            # 异步执行任务
            import asyncio
            asyncio.create_task(_process_async_task(
                task_id, grok_model, messages, "text2image", parameters, auth_token))

            return {
                "output": {
                    "task_id": task_id
                },
                "request_id": task_id
            }
        else:
            # 同步模式：直接返回结果
            request_data = {
                "model": grok_model,
                "messages": messages,
                "stream": False
            }

            result = await GrokClient.openai_to_grok(request_data)

            # 提取图片URL
            if hasattr(result, "choices") and result.choices:
                content = result.choices[0].message.content
                image_urls = re.findall(r'!\[.*?\]\((.*?)\)', content)

                return {
                    "output": {
                        "choices": [{
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": [{
                                    "image": url
                                } for url in image_urls]
                            }
                        }]
                    },
                    "request_id": str(result.id) if hasattr(result, "id") else ""
                }
            else:
                raise HTTPException(status_code=500, detail="生成失败")

    except HTTPException:
        raise
    except GrokApiException as e:
        logger.error(f"[DashScope] Grok API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[DashScope] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal-generation/generation")
async def image2image(
    request: Image2ImageRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        dashscope_security)
):
    """图生图接口（同步模式）"""
    try:
        # 验证认证
        _ = auth_manager.verify(credentials)

        # 提取参数
        messages = request.input.get("messages", [])
        if not messages or not messages[0].get("content"):
            raise HTTPException(status_code=400, detail="缺少messages参数")

        content = messages[0]["content"]

        # 提取图片和文本
        images = []
        prompt = ""

        for item in content:
            if item.get("image"):
                images.append(item["image"])
            elif item.get("text"):
                prompt = item["text"]

        if not images:
            raise HTTPException(status_code=400, detail="缺少图片输入")
        if not prompt:
            raise HTTPException(status_code=400, detail="缺少文本提示词")

        logger.info(f"[DashScope] 从请求中提取到 {len(images)} 张图片")

        # 解析提示词中的图片引用路径（用于验证和匹配）
        import re
        prompt_image_refs = re.findall(
            r'(subjects|scenes)/([^\s\n\)]+\.(?:png|jpg|jpeg|gif|webp))', prompt, re.IGNORECASE)
        logger.info(f"[DashScope] 提示词中引用的图片路径: {prompt_image_refs}")

        # 检查图片数量是否匹配
        if len(prompt_image_refs) > len(images):
            logger.warning(
                f"[DashScope] 警告：提示词中引用了 {len(prompt_image_refs)} 张图片，但只上传了 {len(images)} 张图片")
        elif len(prompt_image_refs) < len(images):
            logger.warning(
                f"[DashScope] 警告：上传了 {len(images)} 张图片，但提示词中只引用了 {len(prompt_image_refs)} 张图片")

        parameters = request.parameters or {}
        size = parameters.get("size", "1280*720")  # 可选，默认1280*720
        n = parameters.get("n", 1)  # 可选，默认1
        prompt_extend = parameters.get("prompt_extend", True)  # 可选，默认true
        watermark = parameters.get("watermark", False)  # 可选，默认false

        # 转换为Grok模型
        grok_model = _convert_to_grok_model(request.model, "image2image")

        # 注意：prompt_extend和watermark参数目前Grok API不支持，但保留在参数中以便兼容

        # 使用官方 API 方式：先上传图片，获取文件 ID，然后在消息中引用图片路径
        from app.services.grok.token import token_manager
        from app.services.grok.upload import ImageUploadManager

        # 获取 token
        token = token_manager.get_token(grok_model)
        if not token:
            raise HTTPException(status_code=500, detail="无法获取认证令牌")

        # 解析提示词中的图片引用路径（用于匹配文件名）
        # 从提示词中提取所有 subjects/xxx.png 和 scenes/xxx.png 格式的引用
        # 保持提示词中的顺序，去重
        prompt_image_refs = re.findall(
            r'(subjects|scenes)/([^\s\n\)]+\.(?:png|jpg|jpeg|gif|webp))', prompt, re.IGNORECASE)

        # 构建期望的文件名列表（保持提示词中的顺序，去重）
        expected_filenames = []
        expected_paths = []  # 保存完整路径（subjects/xxx.png），用于去重
        for prefix, filename in prompt_image_refs:
            full_path = f"{prefix}/{filename}"
            if full_path not in expected_paths:  # 去重，保持第一次出现的顺序
                expected_paths.append(full_path)
                expected_filenames.append(filename)

        logger.info(f"[DashScope] 提示词中引用的图片路径（去重后）: {expected_paths}")
        logger.info(f"[DashScope] 期望的文件名列表（按顺序）: {expected_filenames}")

        # 上传所有图片并获取文件 ID 和文件名
        file_ids = []
        file_names = []
        logger.info(f"[DashScope] 开始上传 {len(images)} 张图片")
        for idx, img_url in enumerate(images):
            logger.info(f"[DashScope] 上传第 {idx+1} 张图片: {img_url[:100]}...")
            file_id, file_uri = await ImageUploadManager.upload(img_url, token)
            if file_id:
                file_ids.append(file_id)
                # 优先使用提示词中引用的文件名，如果没有则从 URL 提取
                import os
                from urllib.parse import urlparse

                if idx < len(expected_filenames):
                    # 使用提示词中引用的文件名（按照顺序匹配）
                    filename = expected_filenames[idx]
                    logger.info(
                        f"[DashScope] 使用提示词中的文件名（第{idx+1}张，匹配到: {expected_paths[idx]}）: {filename}")
                else:
                    # 从 URL 提取文件名
                    if img_url.startswith("http"):
                        parsed = urlparse(img_url)
                        filename = os.path.basename(
                            parsed.path) or f"image_{len(file_names)}.png"
                    elif "data:image" in img_url:
                        # Base64 图片，使用默认名称
                        filename = f"image_{len(file_names)}.png"
                    else:
                        filename = os.path.basename(
                            img_url) or f"image_{len(file_names)}.png"

                file_names.append(filename)
                logger.info(
                    f"[DashScope] 图片 {idx+1} 上传成功: ID={file_id}, 文件名={filename}, URI={file_uri}")
            else:
                logger.warning(
                    f"[DashScope] 图片 {idx+1} 上传失败: {img_url[:100]}...")

        if not file_ids:
            raise HTTPException(status_code=500, detail="图片上传失败")

        logger.info(f"[DashScope] 总共上传 {len(file_ids)} 张图片，文件ID列表: {file_ids}")
        logger.info(f"[DashScope] 实际上传的文件名列表: {file_names}")

        # 验证文件ID与文件名的对应关系
        logger.info(f"[DashScope] 文件ID与文件名的对应关系:")
        for idx, (fid, fname) in enumerate(zip(file_ids, file_names)):
            logger.info(f"[DashScope]   [{idx}] 文件ID: {fid} -> 文件名: {fname}")

        # 验证提示词中的引用是否与实际上传的文件名匹配
        if prompt_image_refs:
            logger.info(f"[DashScope] 验证图片引用匹配:")
            for idx, (prefix, ref_filename) in enumerate(prompt_image_refs):
                if idx < len(file_names):
                    actual_filename = file_names[idx]
                    match_status = "✓" if ref_filename == actual_filename else "✗"
                    logger.info(
                        f"[DashScope]   [{idx}] {match_status} 提示词引用: {prefix}/{ref_filename} -> 实际上传: {actual_filename}")
                else:
                    logger.warning(
                        f"[DashScope]   [{idx}] ✗ 提示词引用了 {prefix}/{ref_filename}，但没有对应的上传图片")

        # 在用户提示词前添加根据图片数量动态生成的前缀
        # 用户传入的提示词已经包含了完整的格式（标题、参考图、描述等），我们只需要添加前缀
        image_count = len(file_ids)
        if image_count == 1:
            prefix_text = "根据上传的图片生成新图，"
        elif image_count == 2:
            prefix_text = "根据上传的两张图生成新图，"
        else:
            prefix_text = f"根据上传的{image_count}张图生成新图，"

        # 保留用户传入的完整提示词（包括标题、参考图列表、描述等），只添加前缀
        enhanced_prompt = f"{prefix_text}{prompt}"

        logger.info(f"[DashScope] 增强后的提示词预览: {enhanced_prompt[:500]}...")

        # 使用官方 API 方式构建请求
        from app.models.grok_models import Models
        grok_model_name, mode = Models.to_grok(grok_model)

        payload = {
            # 图生图使用持久会话
            "temporary": setting.grok_config.get("temporary", False),
            "modelName": grok_model_name,
            "message": enhanced_prompt,
            "fileAttachments": file_ids,  # 使用上传后的文件 ID
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": n,  # 使用请求的 n 参数
            "forceConcise": False,
            "toolOverrides": {"videoGen": False},  # 强制禁用视频生成
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "disableTextFollowUps": False,
            "responseMetadata": {
                "modelConfigOverride": {
                    "modelMap": {}
                },
                "requestModelDetails": {
                    "modelId": grok_model_name
                }
            },
            "disableMemory": False,
            "forceSideBySide": False,
            "modelMode": mode,
            "isAsyncChat": False
        }

        # 直接调用 Grok API（使用官方 API 方式）
        from app.services.grok.client import GrokClient
        from app.services.grok.processer import GrokResponseProcessor
        import asyncio
        import orjson
        from curl_cffi import requests as curl_requests

        # 构建请求头
        from app.services.grok.statsig import get_dynamic_headers
        headers = get_dynamic_headers("/rest/app-chat/conversations/new")
        cf = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{token};{cf}" if cf else token

        proxy = setting.get_proxy("service")
        proxies = {"http": proxy, "https": proxy} if proxy else None

        # 发送请求（带重试机制）
        max_retries = 3
        last_error = None

        for retry in range(max_retries):
            try:
                logger.info(
                    f"[DashScope] 发送请求到 Grok API (尝试 {retry + 1}/{max_retries})")
                response = await asyncio.to_thread(
                    curl_requests.post,
                    "https://grok.com/rest/app-chat/conversations/new",
                    headers=headers,
                    data=orjson.dumps(payload),
                    impersonate="chrome133a",
                    timeout=120,
                    stream=True,
                    proxies=proxies
                )

                if response.status_code != 200:
                    error_msg = f"Grok API 错误: {response.status_code}"
                    try:
                        error_text = response.text[:500] if hasattr(
                            response, 'text') else ""
                        if error_text:
                            error_msg += f" - {error_text}"
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code, detail=error_msg)

                # 处理响应（转换为 OpenAI 格式）
                try:
                    result = await GrokResponseProcessor.process_normal(response, token, grok_model_name)
                    # 成功，跳出重试循环
                    break
                except Exception as process_error:
                    # 如果是网络错误，可以重试
                    error_str = str(process_error)
                    if "HTTP/2" in error_str or "stream" in error_str.lower() or "INTERNAL_ERROR" in error_str:
                        logger.warning(
                            f"[DashScope] 响应处理失败（可能是网络错误）: {process_error}，尝试重试...")
                        if retry < max_retries - 1:
                            await asyncio.sleep(1)  # 等待1秒后重试
                            last_error = process_error
                            continue
                    # 其他错误或重试次数用完，抛出异常
                    raise

            except HTTPException:
                # HTTP 异常直接抛出，不重试
                raise
            except Exception as e:
                error_str = str(e)
                # 如果是网络错误，可以重试
                if ("HTTP/2" in error_str or "stream" in error_str.lower() or
                        "INTERNAL_ERROR" in error_str or "curl" in error_str.lower()):
                    logger.warning(
                        f"[DashScope] 请求失败（可能是网络错误）: {e}，尝试重试 ({retry + 1}/{max_retries})...")
                    if retry < max_retries - 1:
                        await asyncio.sleep(1)  # 等待1秒后重试
                        last_error = e
                        continue
                # 其他错误或重试次数用完
                logger.error(f"[DashScope] 图生图请求失败: {e}")
                raise HTTPException(status_code=500, detail=f"请求失败: {str(e)}")
        else:
            # 所有重试都失败了
            logger.error(
                f"[DashScope] 图生图请求失败（已重试 {max_retries} 次）: {last_error}")
            raise HTTPException(
                status_code=500, detail=f"请求失败（已重试 {max_retries} 次）: {str(last_error)}")

        # 提取图片URL
        if hasattr(result, "choices") and result.choices:
            content = result.choices[0].message.content
            logger.debug(
                f"[DashScope] 多图合成 - Grok原始响应内容类型: {type(content)}, 内容: {str(content)[:500]}")

            # 确保content是字符串
            if not isinstance(content, str):
                logger.warning(
                    f"[DashScope] 多图合成 - content不是字符串类型: {type(content)}")
                content = str(content)

            # 尝试多种方式提取图片URL
            image_urls = []

            # 方法1: Markdown格式 ![](url)
            image_urls = re.findall(r'!\[.*?\]\((.*?)\)', content)

            # 方法2: HTML格式 <img src="url">
            if not image_urls:
                image_urls = re.findall(
                    r'<img[^>]+src=["\'](.*?)["\']', content)

            # 方法3: 直接URL（http://或https://开头，图片格式）
            if not image_urls:
                image_urls = re.findall(
                    r'https?://[^\s<>"\'\)]+\.(?:jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)

            # 方法4: 检查content是否已经是URL格式
            if not image_urls and (content.startswith('http://') or content.startswith('https://')):
                # 检查是否是图片URL
                if any(content.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    image_urls = [content.strip()]

            # 方法5: 如果返回的是视频，说明视频生成被错误启用了，记录警告
            if not image_urls and '<video' in content:
                logger.warning(
                    f"[DashScope] 多图合成返回了视频而不是图片，可能是videoGen被错误启用。内容: {content[:200]}")
                # 尝试从视频URL中提取，虽然这不是正确的行为，但至少能获取到资源
                video_urls = re.findall(
                    r'<video[^>]+src=["\'](.*?)["\']', content)
                if video_urls:
                    logger.warning(
                        f"[DashScope] 检测到视频URL，但期望的是图片URL: {video_urls}")
                    # 不将视频URL添加到image_urls，因为这是错误的

            logger.debug(f"[DashScope] 多图合成 - 提取到的图片URL: {image_urls}")

            if not image_urls:
                logger.warning(
                    f"[DashScope] 多图合成 - 无法提取图片URL，原始内容: {content[:500]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"无法从响应中提取图片URL。响应内容: {content[:200]}"
                )

            # 缓存图片并转换为本地URL
            cached_urls = []
            from app.services.grok.cache import image_cache_service

            # 获取auth_token
            auth_token = None
            if credentials and credentials.credentials:
                from app.services.grok.token import token_manager
                try:
                    token = token_manager.get_token(grok_model)
                    if token:
                        auth_token = token
                except:
                    pass

            for url in image_urls:
                try:
                    # 如果是完整URL，下载并缓存
                    if url.startswith('http://') or url.startswith('https://'):
                        # 提取路径部分用于缓存
                        if '/images/' in url or '/generated/' in url:
                            url_path = url.split(
                                '/images/')[-1] if '/images/' in url else url.split('/generated/')[-1]
                            if not url_path.startswith('/'):
                                url_path = '/' + url_path

                            # 下载并缓存图片
                            cache_path = await image_cache_service.download_image(url_path, auth_token)
                            if cache_path:
                                # 转换为本地访问URL
                                img_path = url_path.replace(
                                    '/', '-').lstrip('-')
                                base_url = setting.global_config.get(
                                    "base_url", "")
                                local_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                cached_urls.append(local_url)
                            else:
                                # 缓存失败，使用原始URL
                                cached_urls.append(url)
                        else:
                            # 不是Grok assets URL，直接使用
                            cached_urls.append(url)
                    else:
                        # 相对路径，直接使用
                        cached_urls.append(url)
                except Exception as e:
                    logger.warning(f"[DashScope] 缓存图片失败: {url} - {e}")
                    cached_urls.append(url)

            return {
                "output": {
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [{
                                "image": url  # 保持原有格式，但确保有URL
                            } for url in cached_urls]
                        }
                    }]
                },
                "request_id": str(result.id) if hasattr(result, "id") else ""
            }
        else:
            raise HTTPException(status_code=500, detail="生成失败：未返回有效响应")

    except HTTPException:
        raise
    except GrokApiException as e:
        logger.error(f"[DashScope] Grok API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[DashScope] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video-generation/video-synthesis")
async def video_synthesis(
    request: Text2VideoRequest | Image2VideoRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        dashscope_security),
    async_header: Optional[str] = Header(None, alias="X-DashScope-Async")
):
    """文生视频/图生视频接口（异步模式）"""
    try:
        # 验证认证
        _ = auth_manager.verify(credentials)

        # 检查是否为异步模式
        is_async = async_header == "enable"

        # 提取参数
        prompt = request.input.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="缺少prompt参数")

        # 判断是文生视频还是图生视频
        img_url = request.input.get("img_url")
        task_type = "image2video" if img_url else "text2video"

        parameters = request.parameters or {}

        if task_type == "text2video":
            # 文生视频：size是必填参数
            size = parameters.get("size")
            if not size:
                raise HTTPException(status_code=400, detail="缺少size参数（必填）")
            resolution = None
        else:
            # 图生视频：resolution是必填参数
            resolution = parameters.get("resolution")
            if not resolution:
                raise HTTPException(
                    status_code=400, detail="缺少resolution参数（必填，支持480P/720P/1080P）")
            if resolution not in ["480P", "720P", "1080P"]:
                raise HTTPException(
                    status_code=400, detail="resolution参数值必须是480P、720P或1080P")
            size = None

        # 可选参数
        prompt_extend = parameters.get("prompt_extend", True)  # 可选，默认true
        audio = parameters.get("audio", True)  # 可选，默认true
        n = parameters.get("n", 1)  # 可选，默认1

        # 注意：prompt_extend和audio参数目前Grok API不支持，但保留在参数中以便兼容

        # 转换为Grok模型
        grok_model = _convert_to_grok_model(request.model, task_type)

        # 构建消息
        images = [img_url] if img_url else None
        messages = _build_grok_messages(prompt, images)

        if is_async:
            # 异步模式：创建任务
            task_id = task_manager.create_task(task_type, {
                "prompt": prompt,
                "img_url": img_url,
                "resolution": resolution,
                "size": size
            })

            # 获取auth_token
            auth_token = None
            if credentials and credentials.credentials:
                from app.services.grok.token import token_manager
                try:
                    token = token_manager.get_token(grok_model)
                    if token:
                        auth_token = token
                except:
                    pass

            # 异步执行任务
            import asyncio
            asyncio.create_task(_process_async_task(
                task_id, grok_model, messages, task_type, parameters, auth_token))

            return {
                "output": {
                    "task_id": task_id
                },
                "request_id": task_id
            }
        else:
            # 同步模式：直接返回结果（不推荐，但支持）
            request_data = {
                "model": grok_model,
                "messages": messages,
                "stream": False
            }

            # 添加视频参数
            if resolution:
                if resolution == "480P":
                    request_data["aspect_ratio"] = "16:9"
                elif resolution == "720P":
                    request_data["aspect_ratio"] = "16:9"
                elif resolution == "1080P":
                    request_data["aspect_ratio"] = "16:9"

            result = await GrokClient.openai_to_grok(request_data)

            # 提取视频URL
            if hasattr(result, "choices") and result.choices:
                content = result.choices[0].message.content
                video_urls = re.findall(
                    r'<video[^>]+src=["\'](.*?)["\']', content)

                return {
                    "output": {
                        "choices": [{
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": [{
                                    "video": url
                                } for url in video_urls]
                            }
                        }]
                    },
                    "request_id": str(result.id) if hasattr(result, "id") else ""
                }
            else:
                raise HTTPException(status_code=500, detail="生成失败")

    except HTTPException:
        raise
    except GrokApiException as e:
        logger.error(f"[DashScope] Grok API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[DashScope] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
