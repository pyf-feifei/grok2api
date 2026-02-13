"""OpenAI 工具调用适配器

将 Grok 文本响应转换为 OpenAI tool_calls 结构，避免在 OpenAI 端点丢失工具协议。
"""

import time
import uuid
import orjson
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
            # 关键修复：不要在模型未生成工具调用时强制“兜底”注入 tool_calls，
            # 否则会导致客户端反复调用同一工具并进入死循环。
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
