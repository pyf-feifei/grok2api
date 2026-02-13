"""OpenAI 工具调用适配器

将 Grok 文本响应转换为 OpenAI tool_calls 结构，避免在 OpenAI 端点丢失工具协议。
"""

import time
import uuid
import orjson
import re
from typing import Any, Dict, List, Optional, AsyncGenerator

from app.core.logger import logger
from app.models.openai_schema import (
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
)
from app.services.anthropic.tool_simulator import ToolSimulator


class OpenAIToolAdapter:
    """OpenAI 工具调用适配器"""

    @staticmethod
    def _extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
        """提取最后一条用户文本，用于工具意图解析上下文"""
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return "\n".join([t for t in texts if t])
        return ""

    @staticmethod
    def _count_context7_failures(messages: List[Dict[str, Any]]) -> int:
        """统计近期 context7 调用失败次数（用于防循环）"""
        count = 0
        for msg in messages or []:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", "") or "").lower()
            if (
                "error searching libraries" in content
                or "fetch failed" in content
                or "context7" in content and "error" in content
            ):
                count += 1
        return count

    @staticmethod
    def _normalize_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """OpenAI tools -> ToolSimulator 可识别格式"""
        if not tools:
            return []

        normalized: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            # OpenAI 格式: {"type":"function","function":{"name":"Read",...}}
            fn = t.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                normalized.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {})
                })
                continue
            # 兼容直接格式: {"name":"Read",...}
            if t.get("name"):
                normalized.append(t)
        return normalized

    @staticmethod
    def _extract_openai_functions(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """提取 OpenAI function 工具定义"""
        functions: List[Dict[str, Any]] = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                functions.append(fn)
        return functions

    @staticmethod
    def _infer_library_name(user_message: str) -> str:
        """从用户消息推断库名（优先 python）"""
        text = (user_message or "").lower()
        if "python" in text:
            return "python"
        if "react" in text:
            return "react"
        if "next" in text:
            return "next.js"
        return "python"

    @classmethod
    def _build_fallback_tool_calls(
        cls,
        tools: Optional[List[Dict[str, Any]]],
        user_message: str
    ) -> List[Dict[str, Any]]:
        """当模型未返回工具调用时，构建兜底 tool_calls（优先 context7 resolve）"""
        functions = cls._extract_openai_functions(tools)
        if not functions:
            return []

        # 优先命中 context7 resolve-library-id
        for fn in functions:
            name = fn.get("name", "")
            lname = name.lower()
            if "context7" in lname and "resolve" in lname:
                args = {
                    "query": user_message or "latest python documentation",
                    "libraryName": cls._infer_library_name(user_message)
                }
                return [{
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": orjson.dumps(args).decode("utf-8")
                    }
                }]

        # 次优：若有 query-docs 且可接受自由 query 参数，则兜底调用
        for fn in functions:
            name = fn.get("name", "")
            lname = name.lower()
            if "context7" in lname and "query" in lname:
                # query-docs 通常要求 libraryId，缺失会失败，不在此强行调用
                continue

        return []

    @staticmethod
    def _to_openai_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """ToolSimulator ToolCall -> OpenAI tool_calls"""
        out: List[Dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls):
            call_id = tc.id if getattr(tc, "id", None) else f"call_{uuid.uuid4().hex[:24]}"
            name = getattr(tc, "name", "")
            args = getattr(tc, "input", {}) or {}
            out.append({
                "index": idx,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": orjson.dumps(args).decode("utf-8")
                }
            })
        return out

    @classmethod
    def adapt_non_stream(
        cls,
        response: OpenAIChatCompletionResponse,
        tools: Optional[List[Dict[str, Any]]],
        messages: List[Dict[str, Any]],
    ) -> OpenAIChatCompletionResponse:
        """非流式响应适配为 OpenAI tool_calls"""
        if not tools or not response.choices:
            return response

        tool_defs = cls._normalize_tools(tools)
        if not tool_defs:
            return response

        user_message = cls._extract_last_user_text(messages)
        original_text = response.choices[0].message.content or ""

        simulator = ToolSimulator(tool_defs)
        cleaned_text, parsed_calls = simulator.parse_response(original_text, user_message)
        if not parsed_calls:
            failure_count = cls._count_context7_failures(messages)
            if failure_count >= 2:
                logger.warning(
                    f"[OpenAI] 检测到 context7 连续失败 {failure_count} 次，跳过兜底工具调用以防循环"
                )
                return response

            fallback_calls = cls._build_fallback_tool_calls(tools, user_message)
            if not fallback_calls:
                return response

            response.choices[0] = OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=fallback_calls
                ),
                finish_reason="tool_calls"
            )
            logger.info(f"[OpenAI] 非流式兜底工具调用: {fallback_calls[0]['function']['name']}")
            return response

        openai_tool_calls = cls._to_openai_tool_calls(parsed_calls)
        response.choices[0] = OpenAIChatCompletionChoice(
            index=0,
            message=OpenAIChatCompletionMessage(
                role="assistant",
                content=cleaned_text or None,
                tool_calls=openai_tool_calls
            ),
            finish_reason="tool_calls"
        )
        logger.info(f"[OpenAI] 非流式工具适配成功: {len(openai_tool_calls)} 个工具调用")
        return response

    @classmethod
    async def adapt_stream(
        cls,
        openai_stream: AsyncGenerator[str, None],
        tools: Optional[List[Dict[str, Any]]],
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """流式响应适配为 OpenAI tool_calls"""
        if not tools:
            async for chunk in openai_stream:
                yield chunk
            return

        tool_defs = cls._normalize_tools(tools)
        if not tool_defs:
            async for chunk in openai_stream:
                yield chunk
            return

        raw_chunks: List[str] = []
        total_text = ""
        first_meta: Dict[str, Any] = {}

        async for chunk in openai_stream:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            if not chunk_str.strip():
                continue
            if "[DONE]" in chunk_str:
                continue

            raw_chunks.append(chunk_str)
            if not chunk_str.startswith("data: "):
                continue
            try:
                payload = orjson.loads(chunk_str[6:].strip())
                if not first_meta:
                    first_meta = {
                        "id": payload.get("id", f"chatcmpl-{uuid.uuid4()}"),
                        "model": payload.get("model", "grok-4-1"),
                        "created": payload.get("created", int(time.time())),
                    }
                choices = payload.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        total_text += content
            except Exception as e:
                logger.warning(f"[OpenAI] 流式工具适配解析失败，跳过 chunk: {e}")
                continue

        user_message = cls._extract_last_user_text(messages)
        simulator = ToolSimulator(tool_defs)
        cleaned_text, parsed_calls = simulator.parse_response(total_text, user_message)

        # 没有工具调用，按原始 SSE 回放，保持兼容
        if not parsed_calls:
            failure_count = cls._count_context7_failures(messages)
            if failure_count >= 2:
                logger.warning(
                    f"[OpenAI] 检测到 context7 连续失败 {failure_count} 次，流式跳过兜底工具调用以防循环"
                )
                for raw in raw_chunks:
                    yield raw
                yield "data: [DONE]\n\n"
                return

            fallback_calls = cls._build_fallback_tool_calls(tools, user_message)
            if fallback_calls:
                meta = first_meta or {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "model": "grok-4-1",
                    "created": int(time.time()),
                }
                # role 起始块
                yield (
                    "data: " + orjson.dumps({
                        "id": meta["id"],
                        "object": "chat.completion.chunk",
                        "created": meta["created"],
                        "model": meta["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }).decode("utf-8") + "\n\n"
                )

                # 仅发一个工具调用，满足“每轮必须工具调用”
                yield (
                    "data: " + orjson.dumps({
                        "id": meta["id"],
                        "object": "chat.completion.chunk",
                        "created": meta["created"],
                        "model": meta["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [fallback_calls[0]]},
                            "finish_reason": None
                        }]
                    }).decode("utf-8") + "\n\n"
                )
                yield (
                    "data: " + orjson.dumps({
                        "id": meta["id"],
                        "object": "chat.completion.chunk",
                        "created": meta["created"],
                        "model": meta["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }]
                    }).decode("utf-8") + "\n\n"
                )
                yield "data: [DONE]\n\n"
                logger.info(f"[OpenAI] 流式兜底工具调用: {fallback_calls[0]['function']['name']}")
                return

            for raw in raw_chunks:
                yield raw
            yield "data: [DONE]\n\n"
            return

        meta = first_meta or {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "model": "grok-4-1",
            "created": int(time.time()),
        }
        openai_tool_calls = cls._to_openai_tool_calls(parsed_calls)

        # 1) role 起始块
        yield (
            "data: " + orjson.dumps({
                "id": meta["id"],
                "object": "chat.completion.chunk",
                "created": meta["created"],
                "model": meta["model"],
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }).decode("utf-8") + "\n\n"
        )

        # 2) 文本块（如果有）
        if cleaned_text:
            yield (
                "data: " + orjson.dumps({
                    "id": meta["id"],
                    "object": "chat.completion.chunk",
                    "created": meta["created"],
                    "model": meta["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": cleaned_text},
                        "finish_reason": None
                    }]
                }).decode("utf-8") + "\n\n"
            )

        # 3) 工具调用块
        for call in openai_tool_calls:
            yield (
                "data: " + orjson.dumps({
                    "id": meta["id"],
                    "object": "chat.completion.chunk",
                    "created": meta["created"],
                    "model": meta["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [call]},
                        "finish_reason": None
                    }]
                }).decode("utf-8") + "\n\n"
            )

        # 4) 结束块
        yield (
            "data: " + orjson.dumps({
                "id": meta["id"],
                "object": "chat.completion.chunk",
                "created": meta["created"],
                "model": meta["model"],
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            }).decode("utf-8") + "\n\n"
        )
        yield "data: [DONE]\n\n"
        logger.info(f"[OpenAI] 流式工具适配成功: {len(openai_tool_calls)} 个工具调用")
