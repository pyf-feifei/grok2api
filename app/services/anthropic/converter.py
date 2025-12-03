"""Anthropic 格式转换器 - 在 Anthropic 和 OpenAI 格式之间转换"""

import time
import uuid
import orjson
from typing import Dict, Any, List, AsyncGenerator, Union, Optional

from app.core.logger import logger
from app.services.anthropic.tool_simulator import ToolSimulator


class AnthropicConverter:
    """Anthropic 和 OpenAI 格式转换器"""

    @staticmethod
    def _extract_system_content(system: Any) -> str:
        """从 system 字段提取文本内容（支持字符串和数组格式）"""
        if system is None:
            return ""

        # 字符串格式
        if isinstance(system, str):
            return system

        # 数组格式（Claude Code 发送的格式）
        if isinstance(system, list):
            texts = []
            for block in system:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        texts.append(block.get("text", ""))
                    elif "text" in block:
                        texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)

        # 其他格式尝试转换为字符串
        return str(system)

    @staticmethod
    def to_openai_format(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """将 Anthropic 请求转换为 OpenAI 格式"""

        # 构建 OpenAI 格式的消息列表
        openai_messages = []

        # 添加系统消息（如果有）- 支持字符串和数组格式
        system = anthropic_request.get("system")
        logger.info(
            f"[Anthropic] 原始请求中的 system 字段: {system} (类型: {type(system)})")
        if system:
            system_content = AnthropicConverter._extract_system_content(system)
            logger.info(
                f"[Anthropic] 提取的系统提示词内容: {system_content[:200] + '...' if system_content and len(system_content) > 200 else system_content}")
            if system_content:
                openai_messages.append({
                    "role": "system",
                    "content": system_content
                })
                logger.info(f"[Anthropic] 已添加系统消息到 OpenAI 格式")
            else:
                logger.warning(f"[Anthropic] 系统字段存在但提取的内容为空")
        else:
            logger.info(f"[Anthropic] 请求中没有 system 字段")

        # 转换消息
        for msg in anthropic_request.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            # Anthropic 的 content 可以是字符串或列表
            if isinstance(content, str):
                openai_messages.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(content, list):
                # 处理多模态内容（支持 Claude Code 的各种内容类型）
                openai_content = []
                tool_calls = []  # 收集工具调用
                tool_results = []  # 收集工具结果

                for block in content:
                    block_type = block.get("type")

                    if block_type == "text":
                        openai_content.append({
                            "type": "text",
                            "text": block.get("text", "")
                        })
                    elif block_type == "image":
                        # Anthropic 图片格式转换
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            media_type = source.get("media_type", "image/jpeg")
                            data = source.get("data", "")
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                        elif source.get("type") == "url":
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": source.get("url", "")
                                }
                            })
                    elif block_type == "thinking":
                        # Claude Code 扩展思考内容 - 转换为普通文本
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            openai_content.append({
                                "type": "text",
                                "text": f"[Thinking]\n{thinking_text}\n[/Thinking]"
                            })
                    elif block_type == "tool_use":
                        # Claude Code 工具调用 - 转换为文本（Grok 不支持工具）
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        tool_id = block.get("id", "")
                        openai_content.append({
                            "type": "text",
                            "text": f"[Tool Call: {tool_name}]\n{orjson.dumps(tool_input).decode()}\n[/Tool Call]"
                        })
                    elif block_type == "tool_result":
                        # Claude Code 工具结果 - 转换为文本
                        tool_use_id = block.get("tool_use_id", "")
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            # 提取文本内容
                            texts = []
                            for item in tool_content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    texts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    texts.append(item)
                            tool_content = "\n".join(texts)
                        openai_content.append({
                            "type": "text",
                            "text": f"[Tool Result]\n{tool_content}\n[/Tool Result]"
                        })
                    # 忽略其他不支持的类型：redacted_thinking, document, search_result 等

                if openai_content:
                    openai_messages.append({
                        "role": role,
                        "content": openai_content
                    })

        # 构建 OpenAI 请求
        openai_request = {
            "model": anthropic_request.get("model"),
            "messages": openai_messages,
            "stream": anthropic_request.get("stream", False),
            "temperature": anthropic_request.get("temperature", 1.0),
            "max_tokens": anthropic_request.get("max_tokens", 4096),
        }

        # 添加可选参数
        if top_p := anthropic_request.get("top_p"):
            openai_request["top_p"] = top_p

        logger.info(f"[Anthropic] 转换请求: {len(openai_messages)} 条消息")

        return openai_request

    @staticmethod
    def to_anthropic_response(
        openai_response: Dict[str, Any],
        model: str,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """将 OpenAI 响应转换为 Anthropic 格式

        Args:
            openai_response: OpenAI 格式的响应
            model: 模型名称
            available_tools: 可用工具列表（用于工具模拟）
        """

        # 提取消息内容
        choices = openai_response.get("choices", [])
        if not choices:
            content_text = ""
            stop_reason = "end_turn"
        else:
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content_text = message.get("content", "")
            finish_reason = first_choice.get("finish_reason", "stop")

            # 映射停止原因
            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "content_filter": "stop_sequence",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        # 提取 token 使用情况（usage 可能为 None）
        usage = openai_response.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) if usage else 0
        output_tokens = usage.get("completion_tokens", 0) if usage else 0

        # 使用工具模拟器处理响应
        content = [{"type": "text", "text": content_text}]

        if available_tools and content_text:
            try:
                simulator = ToolSimulator(available_tools)
                simulated_content = simulator.process_response(content_text)
                if simulated_content:
                    content = simulated_content
                    # 如果有工具调用，修改 stop_reason
                    has_tool_use = any(
                        c.get("type") == "tool_use" for c in content)
                    if has_tool_use:
                        stop_reason = "tool_use"
                        logger.info(
                            f"[Anthropic] 工具模拟: 生成了 {sum(1 for c in content if c.get('type') == 'tool_use')} 个工具调用")
            except Exception as e:
                logger.warning(f"[Anthropic] 工具模拟失败: {e}")

        # 构建 Anthropic 响应
        anthropic_response = {
            "id": openai_response.get("id", f"msg_{uuid.uuid4().hex}"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }

        logger.info(f"[Anthropic] 转换响应: {output_tokens} 个输出 token")

        return anthropic_response

    @staticmethod
    async def to_anthropic_stream(
        openai_stream: AsyncGenerator,
        model: str,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[bytes, None]:
        """将 OpenAI 流式响应转换为 Anthropic 流式格式

        Args:
            openai_stream: OpenAI 流式响应
            model: 模型名称
            available_tools: 可用工具列表（用于工具模拟）
        """

        message_id = f"msg_{uuid.uuid4().hex}"
        created_time = int(time.time())
        content_index = 0
        total_text = ""

        try:
            # 发送 message_start 事件
            start_event = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                }
            }
            yield f"event: message_start\ndata: {orjson.dumps(start_event).decode()}\n\n".encode()

            # 发送 content_block_start 事件
            content_start_event = {
                "type": "content_block_start",
                "index": content_index,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }
            yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event).decode()}\n\n".encode()

            # 处理流式数据
            async for chunk in openai_stream:
                # OpenAI 流式格式: "data: {...}\n\n"
                if isinstance(chunk, bytes):
                    chunk_str = chunk.decode('utf-8')
                else:
                    chunk_str = chunk

                # 跳过空行和 [DONE] 标记
                if not chunk_str.strip() or "[DONE]" in chunk_str:
                    continue

                # 解析 OpenAI SSE 格式
                if chunk_str.startswith("data: "):
                    json_str = chunk_str[6:].strip()

                    try:
                        openai_chunk = orjson.loads(json_str)

                        # 提取内容
                        choices = openai_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                total_text += content

                                # 发送 content_block_delta 事件
                                delta_event = {
                                    "type": "content_block_delta",
                                    "index": content_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content
                                    }
                                }
                                yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event).decode()}\n\n".encode()

                    except Exception as e:
                        logger.warning(f"[Anthropic] 解析流式数据失败: {e}")
                        continue

            # 发送 content_block_stop 事件（文本块结束）
            content_stop_event = {
                "type": "content_block_stop",
                "index": content_index
            }
            yield f"event: content_block_stop\ndata: {orjson.dumps(content_stop_event).decode()}\n\n".encode()

            # 工具模拟：检测代码块并生成工具调用
            tool_calls = []
            stop_reason = "end_turn"

            if available_tools and total_text:
                try:
                    simulator = ToolSimulator(available_tools)
                    _, tool_calls = simulator.parse_response(total_text)

                    if tool_calls:
                        stop_reason = "tool_use"
                        logger.info(
                            f"[Anthropic] 流式工具模拟: 生成了 {len(tool_calls)} 个工具调用")

                        # 为每个工具调用发送事件
                        for tc in tool_calls:
                            content_index += 1

                            # 发送 content_block_start 事件（tool_use）
                            tool_start_event = {
                                "type": "content_block_start",
                                "index": content_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.name,
                                    "input": {}
                                }
                            }
                            yield f"event: content_block_start\ndata: {orjson.dumps(tool_start_event).decode()}\n\n".encode()

                            # 发送 content_block_delta 事件（tool_use input）
                            tool_delta_event = {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": orjson.dumps(tc.input).decode()
                                }
                            }
                            yield f"event: content_block_delta\ndata: {orjson.dumps(tool_delta_event).decode()}\n\n".encode()

                            # 发送 content_block_stop 事件
                            tool_stop_event = {
                                "type": "content_block_stop",
                                "index": content_index
                            }
                            yield f"event: content_block_stop\ndata: {orjson.dumps(tool_stop_event).decode()}\n\n".encode()

                except Exception as e:
                    logger.warning(f"[Anthropic] 流式工具模拟失败: {e}")

            # 发送 message_delta 事件
            delta_event = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None
                },
                "usage": {
                    "output_tokens": len(total_text.split())  # 简单估算
                }
            }
            yield f"event: message_delta\ndata: {orjson.dumps(delta_event).decode()}\n\n".encode()

            # 发送 message_stop 事件
            stop_event = {
                "type": "message_stop"
            }
            yield f"event: message_stop\ndata: {orjson.dumps(stop_event).decode()}\n\n".encode()

            logger.info(
                f"[Anthropic] 流式响应完成: {len(total_text)} 字符, {len(tool_calls)} 个工具调用")

        except Exception as e:
            logger.error(f"[Anthropic] 流式转换错误: {e}")
            # 发送错误事件
            error_event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(e)
                }
            }
            yield f"event: error\ndata: {orjson.dumps(error_event).decode()}\n\n".encode()
