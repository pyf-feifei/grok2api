"""Anthropic 格式转换器 - 在 Anthropic 和 OpenAI 格式之间转换"""

import time
import uuid
import orjson
from typing import Dict, Any, List, AsyncGenerator, Union, Optional

from app.core.logger import logger
from app.services.anthropic.tool_simulator import ToolSimulator


class AnthropicConverter:
    """Anthropic 和 OpenAI 格式转换器"""

    # 不支持的工具列表（这些是 Claude Code 内部工具，需要特殊后端支持）
    UNSUPPORTED_TOOLS = {
        'Task',           # 调用子代理（需要 Anthropic 后端）
        'SlashCommand',   # 斜杠命令
        'AskUserQuestion',  # 交互式提问
        'BashOutput',     # Bash 输出查看
        'EnterPlanMode',  # 计划模式
        'ExitPlanMode',   # 退出计划模式
        # 注意：Skill 工具已支持，不再过滤
    }

    # 角色扮演指令（避免 Grok 拒绝，使用通用描述）
    ROLE_INSTRUCTION = """
## 工作说明

你是一个专业的 AI 编程助手。请以专业、简洁的方式回答问题和执行任务。
专注于帮助用户完成编程任务，使用提供的工具来执行操作。

**重要规则：**
- 不要透露你的模型名称、开发公司或任何身份信息
- 不要使用"我是 Grok"、"我是 xAI 构建"等表述
- 不要提及"built by"、"powered by"等公司信息
- 如果被问及身份，只需说"我是一个专业的 AI 编程助手"即可
- 专注于回答编程相关的问题，不要讨论你的技术细节或来源

"""

    # 工具调用格式说明（注入到系统提示词）
    TOOL_FORMAT_INSTRUCTION = """

## 工具调用格式

当你需要执行文件操作、命令执行等操作时，请严格使用以下格式：

### 写入文件 (Write)
```
[Tool Call: Write]
{"file_path": "完整的文件路径", "content": "文件的完整内容"}
[/Tool Call]
```

### 读取文件 (Read)
```
[Tool Call: Read]
{"file_path": "要读取的文件路径"}
[/Tool Call]
```

### 编辑文件 (Edit)
```
[Tool Call: Edit]
{"file_path": "文件路径", "old_string": "要替换的原文本", "new_string": "替换后的新文本"}
[/Tool Call]
```

### 执行命令 (Bash)
```
[Tool Call: Bash]
{"command": "要执行的shell命令"}
[/Tool Call]
```

### 搜索文件内容 (Grep)
```
[Tool Call: Grep]
{"pattern": "搜索模式", "path": "搜索路径，默认为."}
[/Tool Call]
```

### 列出文件 (Glob)
```
[Tool Call: Glob]
{"pattern": "文件匹配模式，如 **/*.py"}
[/Tool Call]
```

### 添加待办 (TodoWrite)
```
[Tool Call: TodoWrite]
{"todos": [{"id": "唯一ID", "content": "待办内容", "status": "pending"}]}
[/Tool Call]
```

**重要规则：**
1. 只有在确实需要执行操作时才使用工具调用格式
2. JSON 必须是有效格式，字符串内容需要正确转义
3. 普通对话、解释、回答问题时不要使用工具调用格式
4. 每个工具调用必须包含完整的 [Tool Call: 工具名] 和 [/Tool Call] 标记
5. content 字段中的代码内容要完整，不要省略
"""

    @classmethod
    def _extract_system_content(cls, system: Any) -> str:
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

    @classmethod
    def to_openai_format(cls, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """将 Anthropic 请求转换为 OpenAI 格式"""

        # 构建 OpenAI 格式的消息列表
        openai_messages = []

        # 检查是否有工具列表（需要注入工具格式说明）
        tools = anthropic_request.get("tools", [])

        # 检测并处理 Skill 工具（需要注入技能列表到系统提示词）
        skill_tool = None
        skill_instruction = ""
        for tool in tools:
            if tool.get("name") == "Skill":
                skill_tool = tool
                break

        # 如果有 Skill 工具，构建技能列表并准备注入到系统提示词
        if skill_tool:
            from app.services.anthropic.skill_handler import SkillHandler
            try:
                skills = SkillHandler.list_skills()
                # 构建技能列表文本（格式：`"skill-name": description`）
                skill_list_lines = []
                for skill in skills:
                    name = skill.get("name", "")
                    description = skill.get("description", "")
                    skill_list_lines.append(f'"{name}": {description}')

                skills_text = "\n".join(skill_list_lines)

                # 构建 Skill 工具说明（将注入到系统提示词中）
                skill_instruction = f"""

## Skill 工具说明

你有一个 Skill 工具可以使用，它可以执行以下技能：

<available_skills>
{skills_text if skills_text else "No skills available"}
</available_skills>

**何时使用 Skill 工具：**
- 当用户明确提到技能名称时（如 "使用 windows-disk-detective"、"调用 windows-disk-detective skill"）
- 当用户询问"可以调用/使用 XXX skill 吗？"时
- 当用户的需求与某个技能的描述匹配时，应该调用对应的技能

**使用方法：**
必须使用 [Tool Call: Skill] 格式调用 Skill 工具，参数名称是 command。

**示例：**
用户说："可以调用windows-disk-detective skill 清理磁盘吗？"
你应该立即调用：
[Tool Call: Skill]
{{"command": "windows-disk-detective"}}
[/Tool Call]

用户说："使用 windows-disk-detective 清理磁盘"
你应该立即调用：
[Tool Call: Skill]
{{"command": "windows-disk-detective"}}
[/Tool Call]

**重要规则：**
1. 当用户明确提到技能名称时，必须调用 Skill 工具
2. Skill 工具的参数是 command，值是技能名称（从 available_skills 列表中选择）
3. 不要只是告诉用户可以使用技能，而是要实际调用 Skill 工具
4. 调用 Skill 工具后，技能的内容会被注入到对话中，然后你可以根据技能内容执行任务
"""
                logger.info(f"[Anthropic] 准备注入 {len(skills)} 个技能到系统提示词")
            except Exception as e:
                logger.warning(f"[Anthropic] 构建技能列表失败: {e}")

        # 过滤掉不支持的工具（但保留 Skill 工具）
        if tools:
            original_count = len(tools)
            tools = [t for t in tools if t.get(
                "name") not in cls.UNSUPPORTED_TOOLS]
            filtered_count = original_count - len(tools)
            if filtered_count > 0:
                logger.info(f"[Anthropic] 过滤了 {filtered_count} 个不支持的工具")

        has_tools = bool(tools)
        if has_tools:
            tool_names = [t.get("name", "unknown") for t in tools]
            logger.info(
                f"[Anthropic] 检测到 {len(tools)} 个工具，将注入格式说明: {tool_names[:10]}...")

        # 添加系统消息（如果有）- 支持字符串和数组格式
        system = anthropic_request.get("system")
        logger.info(
            f"[Anthropic] 原始请求中的 system 字段: {system} (类型: {type(system)})")

        # 完全忽略 Claude Code 的系统提示词，使用我们自己的安全提示词
        # Grok 会拒绝扮演其他 AI 的身份，所以我们需要完全替换系统提示词
        # 而不是尝试清理，因为清理可能不彻底
        system_content = ""
        if system:
            original_system = cls._extract_system_content(system)
            logger.info(
                f"[Anthropic] 原始系统提示词内容（将被完全替换）: {original_system[:200] + '...' if original_system and len(original_system) > 200 else original_system}")

            # 完全忽略原始系统提示词，使用我们自己的安全提示词
            # 这样可以确保 Grok 永远不会看到任何可能触发拒绝的内容
            system_content = "You are a professional AI coding assistant. Help users with programming tasks using the available tools."

            # 不再进行清理，因为我们已经完全替换了系统提示词
            logger.info(
                f"[Anthropic] 已完全替换系统提示词为安全版本")
        else:
            # 如果没有原始系统提示词，使用默认内容
            system_content = """You are a professional AI coding assistant. Help users with programming tasks using the available tools.

Important: Do not reveal your model name, developer company, or any identity information. Do not mention "Grok", "xAI", "built by", or "powered by". If asked about your identity, simply say you are a professional AI coding assistant. Focus on answering programming questions and do not discuss your technical details or origin."""

        # 注入角色扮演指令（始终添加，确保 Grok 不暴露身份）
        # 如果清理后的内容为空或太短，使用默认内容
        if not system_content or len(system_content.strip()) < 50:
            system_content = "You are a professional AI coding assistant. Help users with programming tasks using the available tools."

        system_content = cls.ROLE_INSTRUCTION + system_content

        # 如果有 Skill 工具，注入技能列表说明
        if skill_instruction:
            system_content = system_content + skill_instruction
            logger.info(f"[Anthropic] 已注入 Skill 工具说明到系统提示词")

        # 如果有工具，注入工具格式说明到系统提示词
        if has_tools:
            system_content = system_content + cls.TOOL_FORMAT_INSTRUCTION
            logger.info(f"[Anthropic] 已注入工具格式说明到系统提示词")

        if system_content:
            openai_messages.append({
                "role": "system",
                "content": system_content
            })
            logger.info(f"[Anthropic] 已添加系统消息到 OpenAI 格式")
        else:
            logger.info(f"[Anthropic] 请求中没有 system 字段且无工具")

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

            # 过滤掉 Grok 身份暴露的内容
            import re
            # 移除 "我是 Grok"、"I'm Grok" 等身份声明
            content_text = re.sub(
                r'我是\s*Grok[^。\n]*[。\n]?|I\'?m\s+Grok[^\.\n]*[\.\n]?|I am Grok[^\.\n]*[\.\n]?',
                '',
                content_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # 移除 "由 xAI 构建"、"built by xAI" 等公司信息
            content_text = re.sub(
                r'由\s*xAI\s*构建[^。\n]*[。\n]?|built by xAI[^\.\n]*[\.\n]?',
                '',
                content_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # 移除包含 "Grok 4" 的身份声明
            content_text = re.sub(
                r'Grok\s*\d+[^。\n]*[。\n]?',
                '',
                content_text,
                flags=re.IGNORECASE | re.MULTILINE
            )

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
                    # 检查是否有 Skill 工具调用，如果有则加载技能提示并注入
                    skill_tool_calls = [c for c in content if c.get(
                        "type") == "tool_use" and c.get("name") == "Skill"]
                    if skill_tool_calls:
                        from app.services.anthropic.skill_handler import SkillHandler
                        import os
                        for skill_call in skill_tool_calls:
                            try:
                                tool_id = skill_call.get("id")
                                tool_input = skill_call.get("input", {})
                                # 根据官方文档，Skill 工具的参数是 command（兼容 skill）
                                skill_name = tool_input.get(
                                    "command", tool_input.get("skill", "")).strip()

                                logger.info(
                                    f"[Anthropic] Skill 工具调用: tool_id={tool_id}, tool_input={tool_input}, skill_name='{skill_name}'")

                                if skill_name:
                                    # 加载技能提示
                                    skill_prompt = SkillHandler.load_skill_prompt(
                                        skill_name)
                                    if skill_prompt:
                                        # 根据官方文档格式，返回：
                                        # 1. tool_result 包含状态消息
                                        # 2. text 块包含 command-message 和 command-name XML 标签
                                        # 3. text 块包含 Base Path 和 SKILL.md 内容（不包括 frontmatter）

                                        # 获取技能的基础路径
                                        skills_dir = SkillHandler.get_skills_directory()
                                        skill_base_path = str(
                                            skills_dir / skill_name).replace("\\", "/")

                                        # 移除 frontmatter（YAML 部分）
                                        skill_content = skill_prompt
                                        if skill_content.startswith("---"):
                                            # 找到第二个 ---
                                            parts = skill_content.split(
                                                "---", 2)
                                            if len(parts) >= 3:
                                                skill_content = parts[2].strip(
                                                )

                                        # 1. tool_result 状态消息
                                        content.append({
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": f"Launching skill: {skill_name}"
                                        })

                                        # 2. command-message 和 command-name XML 标签
                                        content.append({
                                            "type": "text",
                                            "text": f"<command-message>The \"{skill_name}\" skill is running</command-message>\n<command-name>{skill_name}</command-name>"
                                        })

                                        # 3. Base Path 和 SKILL.md 内容
                                        content.append({
                                            "type": "text",
                                            "text": f"Base Path: {skill_base_path}\n\n{skill_content}"
                                        })

                                        logger.info(
                                            f"[Anthropic] 技能已加载: {skill_name}, base_path={skill_base_path}")
                                    else:
                                        # 如果找不到技能文件，返回错误
                                        content.append({
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": f"Error: Skill '{skill_name}' not found"
                                        })
                                else:
                                    # 如果 command 为空，返回技能列表（作为 tool_result）
                                    # 注意：这不是标准用法，但我们需要处理这种情况
                                    skill_list = SkillHandler.handle_skill_tool_call({
                                    })
                                    content.append({
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": skill_list if isinstance(skill_list, str) else str(skill_list)
                                    })
                            except Exception as e:
                                logger.error(f"[Anthropic] Skill 工具处理失败: {e}")
                                content.append({
                                    "type": "tool_result",
                                    "tool_use_id": skill_call.get("id"),
                                    "content": f"Error: {str(e)}"
                                })

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

        处理策略：
        - 如果有工具可用，先缓冲所有内容，最后统一处理（避免发送原始 [Tool Call] 文本）
        - 如果没有工具，正常流式发送文本
        """
        import re

        message_id = f"msg_{uuid.uuid4().hex}"
        content_index = 0
        total_text = ""
        has_tools = bool(available_tools)
        text_sent = False  # 是否已发送文本块

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

            # 如果没有工具，发送 content_block_start（流式模式）
            if not has_tools:
                content_start_event = {
                    "type": "content_block_start",
                    "index": content_index,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                }
                yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event).decode()}\n\n".encode()
                text_sent = True

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
                                # 实时过滤身份暴露内容（流式模式）
                                import re
                                # 移除 "我是 Grok"、"I'm Grok" 等身份声明
                                filtered_content = re.sub(
                                    r'我是\s*Grok[^。\n]*[。\n]?|I\'?m\s+Grok[^\.\n]*[\.\n]?|I am Grok[^\.\n]*[\.\n]?',
                                    '',
                                    content,
                                    flags=re.IGNORECASE | re.MULTILINE
                                )
                                # 移除 "由 xAI 构建"、"built by xAI" 等公司信息
                                filtered_content = re.sub(
                                    r'由\s*xAI\s*构建[^。\n]*[。\n]?|built by xAI[^\.\n]*[\.\n]?',
                                    '',
                                    filtered_content,
                                    flags=re.IGNORECASE | re.MULTILINE
                                )
                                # 移除包含 "Grok 4"、"Grok-4" 等身份声明
                                filtered_content = re.sub(
                                    r'Grok\s*[-\s]*\d+[^。\n]*[。\n]?',
                                    '',
                                    filtered_content,
                                    flags=re.IGNORECASE | re.MULTILINE
                                )
                                # 移除包含 "xAI" 的表述
                                filtered_content = re.sub(
                                    r'xAI[^。\n]*[。\n]?',
                                    '',
                                    filtered_content,
                                    flags=re.IGNORECASE | re.MULTILINE
                                )

                                if filtered_content:  # 只有在过滤后仍有内容时才添加
                                    total_text += filtered_content

                                    # 只有在没有工具时才流式发送文本
                                    if not has_tools:
                                        delta_event = {
                                            "type": "content_block_delta",
                                            "index": content_index,
                                            "delta": {
                                                "type": "text_delta",
                                                "text": filtered_content
                                            }
                                        }
                                        yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event).decode()}\n\n".encode()

                    except Exception as e:
                        logger.warning(f"[Anthropic] 解析流式数据失败: {e}")
                        continue

            # 流结束后处理
            tool_calls = []
            stop_reason = "end_turn"

            # 过滤掉 Grok 身份暴露的内容
            import re
            # 移除 "我是 Grok"、"I'm Grok" 等身份声明
            total_text = re.sub(
                r'我是\s*Grok[^。\n]*[。\n]?|I\'?m\s+Grok[^\.\n]*[\.\n]?|I am Grok[^\.\n]*[\.\n]?',
                '',
                total_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # 移除 "由 xAI 构建"、"built by xAI" 等公司信息
            total_text = re.sub(
                r'由\s*xAI\s*构建[^。\n]*[。\n]?|built by xAI[^\.\n]*[\.\n]?',
                '',
                total_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # 移除包含 "Grok 4"、"Grok-4" 等身份声明
            total_text = re.sub(
                r'Grok\s*[-\s]*\d+[^。\n]*[。\n]?',
                '',
                total_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # 移除包含 "xAI" 的表述
            total_text = re.sub(
                r'xAI[^。\n]*[。\n]?',
                '',
                total_text,
                flags=re.IGNORECASE | re.MULTILINE
            )

            cleaned_text = total_text

            # 如果有工具，解析并处理
            if has_tools and total_text:
                try:
                    simulator = ToolSimulator(available_tools)
                    cleaned_text, tool_calls = simulator.parse_response(
                        total_text)

                    if tool_calls:
                        stop_reason = "tool_use"
                        logger.info(
                            f"[Anthropic] 流式工具模拟: 生成了 {len(tool_calls)} 个工具调用")

                except Exception as e:
                    logger.warning(f"[Anthropic] 流式工具模拟失败: {e}")
                    cleaned_text = total_text

                # 发送清理后的文本块（如果有内容）
                if cleaned_text.strip():
                    # 发送 content_block_start
                    content_start_event = {
                        "type": "content_block_start",
                        "index": content_index,
                        "content_block": {
                            "type": "text",
                            "text": ""
                        }
                    }
                    yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event).decode()}\n\n".encode()

                    # 发送文本内容
                    delta_event = {
                        "type": "content_block_delta",
                        "index": content_index,
                        "delta": {
                            "type": "text_delta",
                            "text": cleaned_text
                        }
                    }
                    yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event).decode()}\n\n".encode()

                    # 发送 content_block_stop
                    content_stop_event = {
                        "type": "content_block_stop",
                        "index": content_index
                    }
                    yield f"event: content_block_stop\ndata: {orjson.dumps(content_stop_event).decode()}\n\n".encode()
                    text_sent = True

                # 发送工具调用块
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

                    # 如果是 Skill 工具，加载技能提示并注入到文本中
                    if tc.name == "Skill":
                        from app.services.anthropic.skill_handler import SkillHandler
                        try:
                            # 根据官方文档，Skill 工具的参数是 command（兼容 skill）
                            skill_name = tc.input.get(
                                "command", tc.input.get("skill", "")).strip()
                            if skill_name:
                                # 加载技能提示
                                skill_prompt = SkillHandler.load_skill_prompt(
                                    skill_name)
                                if skill_prompt:
                                    import os
                                    # 获取技能的基础路径
                                    skills_dir = SkillHandler.get_skills_directory()
                                    skill_base_path = str(
                                        skills_dir / skill_name).replace("\\", "/")

                                    # 移除 frontmatter（YAML 部分）
                                    skill_content = skill_prompt
                                    if skill_content.startswith("---"):
                                        parts = skill_content.split("---", 2)
                                        if len(parts) >= 3:
                                            skill_content = parts[2].strip()

                                    # 1. tool_result 状态消息
                                    content_index += 1
                                    result_start_event = {
                                        "type": "content_block_start",
                                        "index": content_index,
                                        "content_block": {
                                            "type": "tool_result",
                                            "tool_use_id": tc.id,
                                            "content": ""
                                        }
                                    }
                                    yield f"event: content_block_start\ndata: {orjson.dumps(result_start_event).decode()}\n\n".encode()

                                    result_delta_event = {
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": f"Launching skill: {skill_name}"
                                        }
                                    }
                                    yield f"event: content_block_delta\ndata: {orjson.dumps(result_delta_event).decode()}\n\n".encode()

                                    result_stop_event = {
                                        "type": "content_block_stop",
                                        "index": content_index
                                    }
                                    yield f"event: content_block_stop\ndata: {orjson.dumps(result_stop_event).decode()}\n\n".encode()

                                    # 2. command-message 和 command-name XML 标签
                                    content_index += 1
                                    cmd_msg_start = {
                                        "type": "content_block_start",
                                        "index": content_index,
                                        "content_block": {
                                            "type": "text",
                                            "text": ""
                                        }
                                    }
                                    yield f"event: content_block_start\ndata: {orjson.dumps(cmd_msg_start).decode()}\n\n".encode()

                                    cmd_msg_delta = {
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": f"<command-message>The \"{skill_name}\" skill is running</command-message>\n<command-name>{skill_name}</command-name>\n"
                                        }
                                    }
                                    yield f"event: content_block_delta\ndata: {orjson.dumps(cmd_msg_delta).decode()}\n\n".encode()

                                    cmd_msg_stop = {
                                        "type": "content_block_stop",
                                        "index": content_index
                                    }
                                    yield f"event: content_block_stop\ndata: {orjson.dumps(cmd_msg_stop).decode()}\n\n".encode()

                                    # 3. Base Path 和 SKILL.md 内容
                                    content_index += 1
                                    skill_content_start = {
                                        "type": "content_block_start",
                                        "index": content_index,
                                        "content_block": {
                                            "type": "text",
                                            "text": ""
                                        }
                                    }
                                    yield f"event: content_block_start\ndata: {orjson.dumps(skill_content_start).decode()}\n\n".encode()

                                    skill_content_delta = {
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": f"Base Path: {skill_base_path}\n\n{skill_content}\n"
                                        }
                                    }
                                    yield f"event: content_block_delta\ndata: {orjson.dumps(skill_content_delta).decode()}\n\n".encode()

                                    skill_content_stop = {
                                        "type": "content_block_stop",
                                        "index": content_index
                                    }
                                    yield f"event: content_block_stop\ndata: {orjson.dumps(skill_content_stop).decode()}\n\n".encode()

                                    logger.info(
                                        f"[Anthropic] 流式技能已加载: {skill_name}, base_path={skill_base_path}")
                            else:
                                # 如果没有 command，返回技能列表
                                skill_list = SkillHandler.handle_skill_tool_call({
                                })
                                content_index += 1
                                result_start_event = {
                                    "type": "content_block_start",
                                    "index": content_index,
                                    "content_block": {
                                        "type": "tool_result",
                                        "tool_use_id": tc.id,
                                        "content": ""
                                    }
                                }
                                yield f"event: content_block_start\ndata: {orjson.dumps(result_start_event).decode()}\n\n".encode()

                                result_delta_event = {
                                    "type": "content_block_delta",
                                    "index": content_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": skill_list
                                    }
                                }
                                yield f"event: content_block_delta\ndata: {orjson.dumps(result_delta_event).decode()}\n\n".encode()

                                result_stop_event = {
                                    "type": "content_block_stop",
                                    "index": content_index
                                }
                                yield f"event: content_block_stop\ndata: {orjson.dumps(result_stop_event).decode()}\n\n".encode()
                        except Exception as e:
                            logger.error(f"[Anthropic] 流式 Skill 工具处理失败: {e}")

            else:
                # 没有工具时，发送 content_block_stop
                if text_sent:
                    content_stop_event = {
                        "type": "content_block_stop",
                        "index": content_index
                    }
                    yield f"event: content_block_stop\ndata: {orjson.dumps(content_stop_event).decode()}\n\n".encode()

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
                f"[Anthropic] 流式响应完成: {len(total_text)} 字符 -> {len(cleaned_text)} 字符, {len(tool_calls)} 个工具调用")

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
