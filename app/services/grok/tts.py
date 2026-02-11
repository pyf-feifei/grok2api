"""Grok TTS 服务 - 处理文本转语音功能"""

import asyncio
import orjson
from typing import Optional
from curl_cffi import requests as curl_requests

from app.core.config import setting
from app.core.logger import logger
from app.core.exception import GrokApiException
from app.services.grok.token import token_manager
from app.services.grok.statsig import get_dynamic_headers
from app.models.grok_models import Models


# 常量
TTS_AUDIO_ENDPOINT = "https://grok.com/http/app-chat/read-response-audio-file"
CONVERSATION_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
TIMEOUT = 120
BROWSER = "chrome133a"
MAX_RETRY = 3

# 系统提示词：确保AI原样返回文本，不添加任何内容
TTS_SYSTEM_PROMPT = "你是一个文本转语音助手。请原样返回用户输入的内容，不要添加任何解释、标点、格式或额外内容。只返回用户输入的原始文本，一个字都不能改变。"


class GrokTTSService:
    """Grok TTS 服务"""

    @staticmethod
    async def text_to_speech(text: str, model: str = "grok-4.1") -> bytes:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            model: 使用的模型
            
        Returns:
            bytes: 音频文件内容（WAV格式）
        """
        logger.info(f"[TTS] 开始TTS转换，文本长度: {len(text)} 字符")
        
        # 获取token
        token = token_manager.get_token(model)
        if not token:
            raise GrokApiException("认证令牌缺失", "NO_AUTH_TOKEN")
        
        # 重试机制
        last_err = None
        for i in range(MAX_RETRY):
            try:
                # 步骤1: 创建对话，让AI原样返回文本
                response_id = await GrokTTSService._create_conversation_and_get_response_id(
                    text, model, token
                )
                
                if not response_id:
                    raise GrokApiException("无法获取响应ID", "NO_RESPONSE_ID")
                
                logger.info(f"[TTS] 获取到响应ID: {response_id}")
                
                # 步骤2: 通过responseId获取音频文件
                audio_data = await GrokTTSService._get_audio_file(response_id, token)
                
                if not audio_data:
                    raise GrokApiException("无法获取音频文件", "NO_AUDIO_DATA")
                
                logger.info(f"[TTS] 成功获取音频，大小: {len(audio_data)} bytes")
                return audio_data
                
            except GrokApiException as e:
                last_err = e
                # 仅401/429可重试
                if e.error_code not in ["HTTP_ERROR", "NO_AVAILABLE_TOKEN"]:
                    raise
                
                status = e.context.get("status") if e.context else None
                if status not in [401, 429]:
                    raise
                
                if i < MAX_RETRY - 1:
                    logger.warning(f"[TTS] 失败(状态:{status}), 重试 {i+1}/{MAX_RETRY}")
                    await asyncio.sleep(0.5)
        
        raise last_err or GrokApiException("TTS转换失败", "TTS_ERROR")

    @staticmethod
    async def _create_conversation_and_get_response_id(
        text: str, model: str, token: str
    ) -> Optional[str]:
        """
        创建对话并获取responseId
        
        Args:
            text: 要转换的文本
            model: 模型名称
            token: 认证token
            
        Returns:
            Optional[str]: 响应ID
        """
        # 获取模型信息
        grok_model, mode = Models.to_grok(model)
        
        # 构建payload，使用系统提示词确保原样返回
        # 将系统提示词添加到消息开头，确保AI原样返回
        full_message = f"{TTS_SYSTEM_PROMPT}\n\n用户输入：{text}\n\n请原样返回用户输入的内容。"
        
        payload = {
            "temporary": True,
            "modelName": grok_model,
            "message": full_message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": True,  # 禁用搜索，确保原样返回
            "enableImageGeneration": False,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": False,
            "imageGenerationCount": 0,
            "forceConcise": True,  # 强制简洁
            "enableSideBySide": False,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True,
            "responseMetadata": {
                "requestModelDetails": {"modelId": grok_model}
            },
            "disableMemory": True,  # 禁用记忆
            "forceSideBySide": False,
            "modelMode": mode,
            "isAsyncChat": False
        }
        
        # 构建请求头
        headers = get_dynamic_headers("/rest/app-chat/conversations/new")
        cf = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{token};{cf}" if cf else token
        
        proxy = setting.get_proxy("service")
        # 使用和GrokClient完全相同的代理处理逻辑
        proxies = {"http": proxy, "https": proxy} if proxy else None
        
        if proxy:
            logger.debug(f"[TTS] 使用代理: {proxy}")
        else:
            logger.warning("[TTS] 未配置代理，可能导致IP被拦截")
        
        logger.debug(f"[TTS] 发送对话请求，文本: {text[:100]}...")
        
        try:
            # 发送请求
            response = await asyncio.to_thread(
                curl_requests.post,
                CONVERSATION_ENDPOINT,
                headers=headers,
                data=orjson.dumps(payload),
                impersonate=BROWSER,
                timeout=TIMEOUT,
                stream=True,
                proxies=proxies
            )
            
            if response.status_code != 200:
                GrokTTSService._handle_error(response, token)
            
            # 解析响应，提取responseId
            response_id = None
            all_chunks = []  # 保存所有chunk用于调试
            
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                
                try:
                    data = orjson.loads(chunk)
                    all_chunks.append(data)  # 保存用于调试
                    
                    # 错误检查
                    if error := data.get("error"):
                        raise GrokApiException(
                            f"API错误: {error.get('message', '未知错误')}",
                            "API_ERROR",
                            {"code": error.get("code")}
                        )
                    
                    # 提取responseId - 尝试多种可能的路径
                    result_data = data.get("result")
                    if result_data:
                        grok_resp = result_data.get("response", {})
                    else:
                        grok_resp = data.get("response", {})
                    
                    # 方法1: 在modelResponse中查找
                    if model_resp := grok_resp.get("modelResponse"):
                        response_id = (
                            model_resp.get("responseId") or 
                            model_resp.get("id") or
                            model_resp.get("response_id")
                        )
                    
                    # 方法2: 在response的顶层查找
                    if not response_id:
                        response_id = (
                            grok_resp.get("responseId") or 
                            grok_resp.get("id") or
                            grok_resp.get("response_id")
                        )
                    
                    # 方法3: 在result中查找
                    if not response_id and result_data:
                        response_id = (
                            result_data.get("responseId") or 
                            result_data.get("id") or
                            result_data.get("response_id")
                        )
                    
                    # 方法4: 在data的顶层查找
                    if not response_id:
                        response_id = (
                            data.get("responseId") or 
                            data.get("id") or
                            data.get("response_id")
                        )
                    
                    # 方法5: 在userResponse中查找
                    if not response_id:
                        user_resp = grok_resp.get("userResponse", {})
                        response_id = (
                            user_resp.get("responseId") or 
                            user_resp.get("id") or
                            user_resp.get("response_id")
                        )
                    
                    # 如果找到responseId，记录并继续读取（可能需要等待最终响应）
                    if response_id:
                        logger.debug(f"[TTS] 从响应中提取到responseId: {response_id}")
                        # 继续读取，但记录找到的ID
                        # 如果后续有新的ID，使用最新的
                        
                except orjson.JSONDecodeError as e:
                    logger.debug(f"[TTS] JSON解析失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[TTS] 解析响应块时出错: {e}")
                    continue
            
            # 如果还没找到，尝试从所有chunks中深度搜索
            if not response_id:
                logger.warning("[TTS] 常规路径未找到responseId，尝试深度搜索")
                # 将最后一个chunk转换为字符串用于调试
                if all_chunks:
                    try:
                        last_chunk_str = orjson.dumps(all_chunks[-1], option=orjson.OPT_INDENT_2).decode('utf-8')
                        logger.debug(f"[TTS] 最后一个chunk内容（前1000字符）: {last_chunk_str[:1000]}")
                    except:
                        pass
                
                # 深度搜索：查找所有可能的UUID格式字符串
                import re
                uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                for chunk_data in all_chunks:
                    chunk_str = orjson.dumps(chunk_data).decode('utf-8')
                    uuids = re.findall(uuid_pattern, chunk_str, re.IGNORECASE)
                    if uuids:
                        # 使用第一个找到的UUID（可能是responseId）
                        response_id = uuids[0]
                        logger.info(f"[TTS] 通过深度搜索找到可能的responseId: {response_id}")
                        break
            
            response.close()
            
            if not response_id:
                logger.warning("[TTS] 未能从响应中提取responseId，尝试查找所有可能的字段")
                # 如果还是没找到，可能需要重新请求并打印完整响应用于调试
                raise GrokApiException("无法从响应中提取responseId", "NO_RESPONSE_ID")
            
            return response_id
            
        except curl_requests.RequestsError as e:
            logger.error(f"[TTS] 网络错误: {e}")
            raise GrokApiException(f"网络错误: {e}", "NETWORK_ERROR") from e
        except Exception as e:
            logger.error(f"[TTS] 请求错误: {e}")
            raise GrokApiException(f"请求错误: {e}", "REQUEST_ERROR") from e

    @staticmethod
    async def _get_audio_file(response_id: str, token: str) -> Optional[bytes]:
        """
        通过responseId获取音频文件
        
        Args:
            response_id: 响应ID
            token: 认证token
            
        Returns:
            Optional[bytes]: 音频文件内容
        """
        url = f"{TTS_AUDIO_ENDPOINT}/{response_id}"
        
        # 构建请求头
        headers = get_dynamic_headers("/http/app-chat/read-response-audio-file")
        cf = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{token};{cf}" if cf else token
        headers["Referer"] = "https://grok.com/"
        headers["Accept"] = "audio/wav,audio/*,*/*"
        headers["Accept-Encoding"] = "identity"
        headers["Range"] = "bytes=0-"
        
        proxy = setting.get_proxy("service")
        # 如果代理为空字符串，设置为None
        if not proxy or proxy.strip() == "":
            proxies = None
        else:
            proxies = {"http": proxy, "https": proxy}
        
        if proxy:
            logger.debug(f"[TTS] 使用代理获取音频: {proxy}")
        else:
            logger.debug("[TTS] 未使用代理获取音频")
        
        logger.debug(f"[TTS] 请求音频文件: {url}")
        
        try:
            response = await asyncio.to_thread(
                curl_requests.get,
                url,
                headers=headers,
                impersonate=BROWSER,
                timeout=TIMEOUT,
                proxies=proxies
            )
            
            if response.status_code not in [200, 206]:  # 206是Partial Content，也支持
                error_msg = f"获取音频失败: {response.status_code}"
                try:
                    error_text = response.text[:500] if hasattr(response, 'text') else ""
                    if error_text:
                        error_msg += f" - {error_text}"
                except:
                    pass
                logger.error(f"[TTS] {error_msg}")
                raise GrokApiException(error_msg, "AUDIO_FETCH_ERROR")
            
            # 检查Content-Type
            content_type = response.headers.get("Content-Type", "")
            if "audio" not in content_type.lower():
                logger.warning(f"[TTS] 意外的Content-Type: {content_type}")
            
            audio_data = response.content
            logger.info(f"[TTS] 成功获取音频，Content-Type: {content_type}, 大小: {len(audio_data)} bytes")
            
            return audio_data
            
        except curl_requests.RequestsError as e:
            logger.error(f"[TTS] 获取音频网络错误: {e}")
            raise GrokApiException(f"获取音频网络错误: {e}", "NETWORK_ERROR") from e
        except Exception as e:
            logger.error(f"[TTS] 获取音频错误: {e}")
            raise GrokApiException(f"获取音频错误: {e}", "AUDIO_ERROR") from e

    @staticmethod
    def _handle_error(response, token: str):
        """处理错误响应 - 透传上游原始错误信息"""
        try:
            data = response.json()
            msg = str(data)
        except Exception:
            try:
                raw = response.text
                data = raw[:500] if raw else ""
                msg = data[:200] if data else "未知错误"
            except Exception:
                data = ""
                msg = "未知错误"

        if response.status_code == 403:
            is_cf_block = not isinstance(data, dict)
            if is_cf_block:
                cf_msg = "您的IP被拦截，请尝试以下方法之一: 1.更换IP 2.使用代理 3.配置CF值"
                logger.warning(f"[TTS] {cf_msg} (原始响应: {msg[:200]})")
                data = {"cf_blocked": True, "status": 403, "upstream_detail": msg[:200]}
                msg = cf_msg
            else:
                logger.warning(f"[TTS] 上游403错误: {msg}")

        asyncio.create_task(token_manager.record_failure(
            token, response.status_code, msg))
        raise GrokApiException(
            f"请求失败: {response.status_code} - {msg}",
            "HTTP_ERROR",
            {"status": response.status_code, "data": data}
        )

