"""Grok API 客户端 - 处理OpenAI到Grok的请求转换和响应处理"""

import asyncio
import orjson
from typing import Dict, List, Tuple, Any, Optional
from curl_cffi import requests as curl_requests

from app.core.config import setting
from app.core.logger import logger
from app.models.grok_models import Models
from app.services.grok.processer import GrokResponseProcessor
from app.services.grok.statsig import get_dynamic_headers
from app.services.grok.token import token_manager
from app.services.grok.upload import ImageUploadManager
from app.services.grok.create import PostCreateManager
from app.core.exception import GrokApiException


# 常量
API_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
TIMEOUT = 120
BROWSER = "chrome133a"
MAX_RETRY = 3
MAX_UPLOADS = 5
MAX_CONCURRENT_REQUESTS = 1  # 限制并发请求数为 1，防止 Grok API 的 403 限制
REQUEST_DELAY = 0.5  # 请求之间的延迟（秒）


class GrokClient:
    """Grok API 客户端"""

    _upload_sem = asyncio.Semaphore(MAX_UPLOADS)
    _request_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # 全局请求信号量
    _last_request_time = 0.0  # 上次请求时间

    @staticmethod
    async def openai_to_grok(request: dict):
        """转换OpenAI请求为Grok请求"""
        # 打印上传的原始请求内容
        logger.info("=" * 80)
        logger.info("[Client] ========== 上传的原始请求内容 ==========")
        logger.info(f"[Client] 模型: {request.get('model')}")
        logger.info(f"[Client] 流式: {request.get('stream', False)}")
        logger.info(f"[Client] 消息数量: {len(request.get('messages', []))}")
        logger.info(f"[Client] 完整消息列表:")
        for idx, msg in enumerate(request.get('messages', [])):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if isinstance(content, str):
                # 截断过长的内容
                content_preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"[Client]   [{idx}] {role}: {content_preview}")
            elif isinstance(content, list):
                logger.info(f"[Client]   [{idx}] {role}: [多部分内容，共{len(content)}项]")
                for part_idx, part in enumerate(content):
                    part_type = part.get('type', 'unknown')
                    if part_type == 'text':
                        text_preview = part.get('text', '')[:200] + "..." if len(part.get('text', '')) > 200 else part.get('text', '')
                        logger.info(f"[Client]     - {part_type}: {text_preview}")
                    elif part_type == 'image_url':
                        img_url = part.get('image_url', {}).get('url', '')[:100] + "..." if len(part.get('image_url', {}).get('url', '')) > 100 else part.get('image_url', {}).get('url', '')
                        logger.info(f"[Client]     - {part_type}: {img_url}")
            else:
                logger.info(f"[Client]   [{idx}] {role}: {type(content)}")
        logger.info("=" * 80)
        
        model = request["model"]
        content, images, system_prompt = GrokClient._extract_content(
            request["messages"])
        
        # 打印提取后的内容
        logger.info("[Client] ========== 提取后的内容 ==========")
        logger.info(f"[Client] 合并后的文本内容长度: {len(content)} 字符")
        content_preview = content[:1000] + "..." if len(content) > 1000 else content
        logger.info(f"[Client] 合并后的文本内容预览: {content_preview}")
        logger.info(f"[Client] 图片数量: {len(images)}")
        if images:
            for idx, img in enumerate(images):
                img_preview = img[:100] + "..." if len(img) > 100 else img
                logger.info(f"[Client]   [{idx}] {img_preview}")
        logger.info(f"[Client] 系统提示词: {system_prompt[:200] + '...' if system_prompt and len(system_prompt) > 200 else system_prompt}")
        logger.info("=" * 80)
        
        stream = request.get("stream", False)
        aspect_ratio = request.get("aspect_ratio")
        # 优先使用 duration（OpenAI 官方格式），如果没有则使用 video_length（兼容）
        video_length = request.get("duration") or request.get("video_length")
        # 是否强制禁用视频生成（仅用于 DashScope 图生图接口，不影响正常聊天、Gemini、文生图、图生视频等接口）
        force_disable_video = request.get("force_disable_video", False)

        # 获取模型信息
        info = Models.get_model_info(model)
        grok_model, mode = Models.to_grok(model)
        is_video = info.get("is_video_model", False) and not force_disable_video

        # 视频模型支持多张图片（不再限制为1张）
        if is_video and len(images) > 1:
            logger.info(f"[Client] 视频模型支持多张图片，共 {len(images)} 张")

        return await GrokClient._retry(model, content, images, grok_model, mode, is_video, stream, aspect_ratio, video_length, system_prompt, force_disable_video)

    @staticmethod
    async def _retry(model: str, content: str, images: List[str], grok_model: str, mode: str, is_video: bool, stream: bool, aspect_ratio: Optional[str] = None, video_length: Optional[int] = None, system_prompt: Optional[str] = None, force_disable_video: bool = False):
        """重试请求"""
        last_err = None

        for i in range(MAX_RETRY):
            try:
                token = token_manager.get_token(model)
                img_ids, img_uris = await GrokClient._upload(images, token)

                # 视频模型创建会话（为所有图片创建 post，但只使用第一个作为 parentPostId）
                post_ids = []
                if is_video and img_ids and img_uris:
                    logger.info(f"[Client] ========== 开始为 {len(images)} 张图片创建 post ==========")
                    for idx, (img_id, img_uri) in enumerate(zip(img_ids, img_uris)):
                        # 获取对应的原始图片 URL
                        original_url = images[idx] if idx < len(images) else "未知"
                        
                        post_id = await GrokClient._create_post(img_id, img_uri, token)
                        if post_id:
                            post_ids.append(post_id)
                            # 判断是首帧还是尾帧
                            if len(img_ids) == 1:
                                frame_type = "单帧"
                            elif idx == 0:
                                frame_type = "首帧"
                            elif idx == len(img_ids) - 1:
                                frame_type = "尾帧"
                            else:
                                frame_type = f"第{idx+1}帧"
                            
                            # 截断过长的 URL
                            url_preview = original_url[:100] + "..." if len(original_url) > 100 else original_url
                            logger.info(f"[Client] {frame_type} - Post ID: {post_id}")
                            logger.info(f"[Client] {frame_type} - 图片 URL: {url_preview}")
                    
                    logger.info(f"[Client] 为 {len(img_ids)} 张图片创建了 {len(post_ids)} 个 post")
                    if len(post_ids) >= 2:
                        logger.info(f"[Client] ========== 首尾帧对应关系 ==========")
                        logger.info(f"[Client] 首帧 Post ID: {post_ids[0]}")
                        first_url = images[0] if len(images) > 0 else "未知"
                        logger.info(f"[Client] 首帧图片: {first_url[:100]}...")
                        logger.info(f"[Client] 尾帧 Post ID: {post_ids[-1]}")
                        last_url = images[-1] if len(images) > 0 else "未知"
                        logger.info(f"[Client] 尾帧图片: {last_url[:100]}...")

                payload = GrokClient._build_payload(
                    content, grok_model, mode, img_ids, img_uris, is_video, post_ids, aspect_ratio, video_length, system_prompt, force_disable_video)
                # 使用第一个 post_id 作为 referer
                first_post_id = post_ids[0] if post_ids else None
                return await GrokClient._request(payload, token, model, stream, first_post_id)

            except GrokApiException as e:
                last_err = e
                # 仅 HTTP_ERROR 和 NO_AVAILABLE_TOKEN 可重试
                if e.error_code not in ["HTTP_ERROR", "NO_AVAILABLE_TOKEN"]:
                    raise

                status = e.context.get("status") if e.context else None
                # 401/429/403 都可以重试（403 可能是临时的 IP 限制）
                if status not in [401, 429, 403]:
                    raise

                if i < MAX_RETRY - 1:
                    # 403 需要更长的等待时间
                    wait_time = 2.0 if status == 403 else 0.5
                    logger.warning(
                        f"[Client] 失败(状态:{status}), 重试 {i+1}/{MAX_RETRY}，等待 {wait_time}s")
                    await asyncio.sleep(wait_time)

        raise last_err or GrokApiException("请求失败", "REQUEST_ERROR")

    @staticmethod
    def _extract_content(messages: List[Dict]) -> Tuple[str, List[str], Optional[str]]:
        """提取文本、图片和系统提示词"""
        texts, images = [], []
        system_prompt = None

        logger.info(f"[Client] 开始提取内容，消息数量: {len(messages)}")
        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            logger.debug(f"[Client] 消息 [{idx}] role={role}, content类型={type(content)}, content长度={len(str(content)) if content else 0}")

            # 提取系统提示词（只取第一个 system 消息）
            if role == "system" and system_prompt is None:
                logger.info(f"[Client] 发现系统消息，开始提取系统提示词")
                if isinstance(content, str):
                    system_prompt = content
                    logger.info(f"[Client] 系统提示词（字符串）: {system_prompt[:200] + '...' if len(system_prompt) > 200 else system_prompt}")
                elif isinstance(content, list):
                    # 系统提示词通常只有文本，但也要处理列表格式
                    logger.info(f"[Client] 系统提示词是列表格式，包含 {len(content)} 项")
                    for item_idx, item in enumerate(content):
                        if item.get("type") == "text":
                            system_prompt = item.get("text", "")
                            logger.info(f"[Client] 从列表项 [{item_idx}] 提取系统提示词: {system_prompt[:200] + '...' if len(system_prompt) > 200 else system_prompt}")
                            break
                    if not system_prompt:
                        logger.warning(f"[Client] 系统消息是列表格式但未找到 text 类型的内容")
                else:
                    logger.warning(f"[Client] 系统消息的内容类型不支持: {type(content)}")
                continue  # 系统消息不添加到用户消息中

            # 提取用户和助手消息的文本和图片
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        if url := item.get("image_url", {}).get("url"):
                            images.append(url)
            else:
                texts.append(content)

        return "".join(texts), images, system_prompt

    @staticmethod
    async def _upload(urls: List[str], token: str) -> Tuple[List[str], List[str]]:
        """并发上传图片"""
        if not urls:
            return [], []

        async def upload_limited(url):
            async with GrokClient._upload_sem:
                return await ImageUploadManager.upload(url, token)

        results = await asyncio.gather(*[upload_limited(u) for u in urls], return_exceptions=True)

        ids, uris = [], []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning(f"[Client] 上传失败: {url} - {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                fid, furi = result
                if fid:
                    ids.append(fid)
                    uris.append(furi)

        return ids, uris

    @staticmethod
    async def _create_post(file_id: str, file_uri: str, token: str) -> Optional[str]:
        """创建视频会话"""
        try:
            result = await PostCreateManager.create(file_id, file_uri, token)
            if result and result.get("success"):
                return result.get("post_id")
        except Exception as e:
            logger.warning(f"[Client] 创建会话失败: {e}")
        return None

    @staticmethod
    def _build_payload(content: str, model: str, mode: str, img_ids: List[str], img_uris: List[str], is_video: bool = False, post_ids: List[str] = None, aspect_ratio: Optional[str] = None, video_length: Optional[int] = None, system_prompt: Optional[str] = None, force_disable_video: bool = False) -> Dict:
        """构建请求载荷"""
        # 如果有系统提示词，将其添加到消息开头
        if system_prompt:
            # 将系统提示词和用户消息组合
            full_message = f"{system_prompt}\n\n{content}" if content else system_prompt
        else:
            full_message = content

        # 视频模型特殊处理
        if is_video and img_uris:
            # 构建多个图片 URL（支持多张图片）
            img_urls = []
            if post_ids:
                # 如果有 post_ids，使用 grok.com/imagine/{post_id} 格式
                for post_id in post_ids:
                    img_urls.append(f"https://grok.com/imagine/{post_id}")
            else:
                # 如果没有 post_ids，使用 assets.grok.com/post/{uri} 格式
                for img_uri in img_uris:
                    img_urls.append(f"https://assets.grok.com/post/{img_uri}")
            
            # 将所有图片 URL 用空格连接
            img_msg = " ".join(img_urls)

            # 检查用户提示词中是否已包含 --mode 参数
            import re
            mode_pattern = r'--mode=\w+'
            has_mode = bool(re.search(mode_pattern, full_message))

            # 如果用户没有指定 mode，默认使用 custom
            if not has_mode:
                message = f"{img_msg}  {full_message} --mode=custom"
            else:
                message = f"{img_msg}  {full_message}"

            payload = {
                "temporary": True,
                "modelName": "grok-3",
                "message": message,
                "fileAttachments": img_ids,
                "toolOverrides": {"videoGen": True} if not force_disable_video else {}
            }

            # 添加 responseMetadata（如果提供了视频参数或 fileAttachments 不为空）
            if img_ids:
                response_metadata = {
                    "modelConfigOverride": {
                        "modelMap": {
                            "videoGenModelConfig": {
                                # 使用第一个 post_id 作为 parentPostId（如果有多张图片，使用第一张的 post_id）
                                "parentPostId": post_ids[0] if post_ids else img_ids[0]
                            }
                        }
                    }
                }

                # 添加可选的视频参数
                if aspect_ratio:
                    response_metadata["modelConfigOverride"]["modelMap"]["videoGenModelConfig"]["aspectRatio"] = aspect_ratio
                if video_length:
                    response_metadata["modelConfigOverride"]["modelMap"]["videoGenModelConfig"]["videoLength"] = video_length

                payload["responseMetadata"] = response_metadata

            # 打印视频模型的 payload
            logger.info("[Client] ========== 处理后发送给 Grok API 的内容 (视频模型) ==========")
            logger.info(f"[Client] 接口: {API_ENDPOINT}")
            logger.info(f"[Client] 模型: {payload.get('modelName')}")
            logger.info(f"[Client] 消息长度: {len(payload.get('message', ''))} 字符")
            message_preview = payload.get('message', '')[:1000] + "..." if len(payload.get('message', '')) > 1000 else payload.get('message', '')
            logger.info(f"[Client] 消息内容预览: {message_preview}")
            logger.info(f"[Client] 文件附件数量: {len(payload.get('fileAttachments', []))}")
            if payload.get('fileAttachments'):
                for idx, fid in enumerate(payload.get('fileAttachments', [])):
                    frame_type = "首帧" if idx == 0 else f"第{idx+1}帧" if idx < len(payload.get('fileAttachments', [])) - 1 else "尾帧"
                    logger.info(f"[Client]   [{idx}] {frame_type} - 文件ID: {fid}")
            
            # 显示首尾帧对应关系
            if post_ids and len(post_ids) >= 2:
                logger.info(f"[Client] ========== 首尾帧对应关系 ==========")
                logger.info(f"[Client] 首帧 Post ID: {post_ids[0]} (对应 message 中第一个图片 URL)")
                logger.info(f"[Client] 尾帧 Post ID: {post_ids[-1]} (对应 message 中最后一个图片 URL)")
                logger.info(f"[Client] parentPostId (使用首帧): {payload.get('responseMetadata', {}).get('modelConfigOverride', {}).get('modelMap', {}).get('videoGenModelConfig', {}).get('parentPostId')}")
            logger.info(f"[Client] 完整 Payload (JSON):")
            try:
                payload_json = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode('utf-8')
                logger.info(f"[Client] {payload_json}")
            except Exception as e:
                logger.warning(f"[Client] 无法序列化 payload: {e}")
                logger.info(f"[Client] {payload}")
            logger.info("=" * 80)

            return payload

        # 标准载荷
        payload = {
            "temporary": setting.grok_config.get("temporary", True),
            "modelName": model,
            "message": full_message,
            "fileAttachments": img_ids,
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True,
            "responseMetadata": {"requestModelDetails": {"modelId": model}},
            "disableMemory": False,
            "forceSideBySide": False,
            "modelMode": mode,
            "isAsyncChat": False
        }
        
        # 只有在 DashScope 图生图接口需要禁用视频生成时才添加 toolOverrides
        # 注意：正常聊天、Gemini、文生图、图生视频等接口不会传递 force_disable_video，因此不受影响
        if force_disable_video:
            payload["toolOverrides"] = {"videoGen": False}
        
        # 打印构建的 payload（发送给 Grok API 的内容）
        logger.info("[Client] ========== 处理后发送给 Grok API 的内容 ==========")
        logger.info(f"[Client] 接口: {API_ENDPOINT}")
        logger.info(f"[Client] 模型: {payload.get('modelName')}")
        logger.info(f"[Client] 模式: {payload.get('modelMode')}")
        logger.info(f"[Client] 临时会话: {payload.get('temporary')}")
        logger.info(f"[Client] 消息长度: {len(payload.get('message', ''))} 字符")
        message_preview = payload.get('message', '')[:1000] + "..." if len(payload.get('message', '')) > 1000 else payload.get('message', '')
        logger.info(f"[Client] 消息内容预览: {message_preview}")
        logger.info(f"[Client] 文件附件数量: {len(payload.get('fileAttachments', []))}")
        if payload.get('fileAttachments'):
            for idx, fid in enumerate(payload.get('fileAttachments', [])):
                logger.info(f"[Client]   [{idx}] 文件ID: {fid}")
        logger.info(f"[Client] 完整 Payload (JSON):")
        try:
            payload_json = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode('utf-8')
            logger.info(f"[Client] {payload_json}")
        except Exception as e:
            logger.warning(f"[Client] 无法序列化 payload: {e}")
            logger.info(f"[Client] {payload}")
        logger.info("=" * 80)
        
        return payload

    @staticmethod
    async def _request(payload: dict, token: str, model: str, stream: bool, post_id: str = None):
        """发送请求（带并发限制）"""
        if not token:
            raise GrokApiException("认证令牌缺失", "NO_AUTH_TOKEN")

        # 使用信号量限制并发请求数
        async with GrokClient._request_sem:
            # 确保请求之间有足够的间隔，防止触发 Grok API 的 403 限制
            import time
            current_time = time.time()
            time_since_last = current_time - GrokClient._last_request_time
            if time_since_last < REQUEST_DELAY:
                wait_time = REQUEST_DELAY - time_since_last
                logger.debug(f"[Client] 等待 {wait_time:.2f}s 以避免并发限制...")
                await asyncio.sleep(wait_time)
            
            GrokClient._last_request_time = time.time()
            logger.debug(f"[Client] 获取到请求锁，开始发送请求")

        try:
            # 构建请求
            headers = GrokClient._build_headers(token)
            if model == "grok-imagine-0.9":
                file_attachments = payload.get("fileAttachments") or [""]
                ref_id = post_id or (
                    file_attachments[0] if file_attachments else "")
                if ref_id:
                    headers["Referer"] = f"https://grok.com/imagine/{ref_id}"

            proxy = setting.get_proxy("service")
            proxies = {"http": proxy, "https": proxy} if proxy else None

            if proxy:
                logger.debug(f"[Client] 使用代理: {proxy}")
            else:
                logger.warning("[Client] 未配置代理，可能导致IP被拦截")

            # 执行请求
            try:
                response = await asyncio.to_thread(
                    curl_requests.post,
                    API_ENDPOINT,
                    headers=headers,
                    data=orjson.dumps(payload),
                    impersonate=BROWSER,
                    timeout=TIMEOUT,
                    stream=True,
                    proxies=proxies
                )
            except Exception as e:
                logger.error(f"[Client] 请求异常: {e}, 代理: {proxy}")
                raise

            if response.status_code != 200:
                GrokClient._handle_error(response, token)

            # 成功 - 重置失败计数
            asyncio.create_task(token_manager.reset_failure(token))

            # 处理响应
            result = (GrokResponseProcessor.process_stream(response, token) if stream
                      else await GrokResponseProcessor.process_normal(response, token, model))

            asyncio.create_task(GrokClient._update_limits(token, model))
            return result

        except curl_requests.RequestsError as e:
            logger.error(f"[Client] 网络错误: {e}")
            raise GrokApiException(f"网络错误: {e}", "NETWORK_ERROR") from e
        except GrokApiException:
            # 已经是 GrokApiException，直接重新抛出，保留原始 error_code
            raise
        except Exception as e:
            logger.error(f"[Client] 请求错误: {e}")
            raise GrokApiException(f"请求错误: {e}", "REQUEST_ERROR") from e

    @staticmethod
    def _build_headers(token: str) -> Dict[str, str]:
        """构建请求头"""
        headers = get_dynamic_headers("/rest/app-chat/conversations/new")
        cf = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{token};{cf}" if cf else token
        return headers

    @staticmethod
    def _handle_error(response, token: str):
        """处理错误"""
        if response.status_code == 403:
            # 尝试获取更详细的错误信息
            try:
                error_detail = response.text[:500] if hasattr(
                    response, 'text') else ""
                logger.debug(f"[Client] 403错误详情: {error_detail}")
            except:
                pass
            msg = "您的IP被拦截，请尝试以下方法之一: 1.更换IP 2.使用代理 3.配置CF值"
            data = {"cf_blocked": True, "status": 403}
            logger.warning(f"[Client] {msg}")
        else:
            try:
                data = response.json()
                msg = str(data)
            except:
                data = response.text
                msg = data[:200] if data else "未知错误"

        asyncio.create_task(token_manager.record_failure(
            token, response.status_code, msg))
        raise GrokApiException(
            f"请求失败: {response.status_code} - {msg}",
            "HTTP_ERROR",
            {"status": response.status_code, "data": data}
        )

    @staticmethod
    async def _update_limits(token: str, model: str):
        """更新速率限制"""
        try:
            await token_manager.check_limits(token, model)
        except Exception as e:
            logger.error(f"[Client] 更新限制失败: {e}")
