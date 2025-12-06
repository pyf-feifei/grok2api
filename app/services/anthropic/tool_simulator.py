"""工具调用模拟器 - 将 Grok 的纯文本响应转换为 Anthropic tool_use 格式

此模块解析 Grok 返回的文本响应，识别代码块和文件操作意图，
并将其转换为 Claude Code 可以执行的 tool_use 格式。
"""

import re
import uuid
import orjson
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.core.logger import logger


@dataclass
class CodeBlock:
    """代码块数据类"""
    language: str
    content: str
    file_path: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ToolCall:
    """工具调用数据类"""
    id: str
    name: str
    input: Dict[str, Any]


class ToolSimulator:
    """工具调用模拟器

    解析 Grok 响应，检测代码块和文件操作意图，生成 tool_use 格式。
    支持的工具：
    - Write: 写入新文件
    - Edit: 编辑现有文件（搜索替换）
    - Bash: 执行命令
    """

    # 代码块正则：```language\n...code...\n```
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w+)?\n(.*?)```',
        re.DOTALL
    )

    # Grok 模拟的工具调用格式：[Tool Call: Write]\n{"file_path":"...","content":"..."}\n[/Tool Call]
    # 使用更智能的匹配：先找到开始标记，然后使用平衡括号匹配完整的 JSON
    GROK_TOOL_CALL_PATTERN = re.compile(
        r'\[Tool Call:\s*(\w+)\]\s*\n?',
        re.IGNORECASE
    )

    # 更宽松的格式：带代码块的工具调用
    GROK_TOOL_CALL_PATTERN_ALT = re.compile(
        r'\[Tool Call:\s*(\w+)\]\s*\n?```(?:json)?\s*\n?',
        re.IGNORECASE
    )

    # 结束标记
    TOOL_CALL_END_PATTERN = re.compile(
        r'\s*\n?\[/Tool Call\]',
        re.IGNORECASE
    )

    # 文件路径模式
    FILE_PATH_PATTERNS = [
        # "创建文件 path/to/file.py" 或 "创建 path/to/file.py"
        re.compile(
            r'创建(?:文件)?\s*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "写入 path/to/file.py" 或 "写入到 path/to/file.py"
        re.compile(
            r'写入(?:到)?\s*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "保存为 path/to/file.py" 或 "保存到 path/to/file.py"
        re.compile(
            r'保存(?:为|到)?\s*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "文件 path/to/file.py" 或 "文件: path/to/file.py"
        re.compile(
            r'文件[:\s]*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "Create file path/to/file.py"
        re.compile(
            r'create\s+(?:file\s+)?[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "Write to path/to/file.py"
        re.compile(
            r'write\s+(?:to\s+)?[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "Save as path/to/file.py"
        re.compile(
            r'save\s+(?:as\s+)?[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "File: path/to/file.py" 或 "File path/to/file.py"
        re.compile(
            r'file[:\s]+[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # 路径格式：src/xxx.py, ./xxx.py, path/to/file.ext
        re.compile(r'[`"\']([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)*\.\w+)[`"\']'),
        # 代码块前的注释：# filename.py 或 // filename.py
        re.compile(r'[#/]+\s*([a-zA-Z0-9_\-./\\]+\.\w+)\s*$', re.MULTILINE),
    ]

    # Bash 命令模式 - 匹配明确的执行命令意图
    BASH_PATTERNS = [
        # "运行命令 xxx" 或 "执行 xxx"
        re.compile(
            r'(?:运行|执行)(?:命令|以下命令)?[:\s]*[`"\']([^`"\']+)[`"\']', re.IGNORECASE),
        # "Run: xxx" 或 "Execute: xxx"
        re.compile(
            r'(?:run|execute)[:\s]+[`"\']([^`"\']+)[`"\']', re.IGNORECASE),
    ]

    # Read 文件模式
    READ_PATTERNS = [
        # "读取文件 xxx" 或 "查看 xxx"
        re.compile(
            r'(?:读取|查看|打开)(?:文件)?[:\s]*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "Read file xxx" 或 "View xxx"
        re.compile(
            r'(?:read|view|open|cat)\s+(?:file\s+)?[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
    ]

    # Edit 文件模式
    EDIT_PATTERNS = [
        # "修改文件 xxx" 或 "编辑 xxx"
        re.compile(
            r'(?:修改|编辑|更新)(?:文件)?[:\s]*[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
        # "Edit file xxx" 或 "Modify xxx"
        re.compile(
            r'(?:edit|modify|update)\s+(?:file\s+)?[`"\']?([a-zA-Z0-9_\-./\\]+\.\w+)[`"\']?', re.IGNORECASE),
    ]

    # Grep 搜索模式 - 更宽松的匹配
    GREP_PATTERNS = [
        # "搜索 xxx" 或 "查找 xxx"
        re.compile(
            r'(?:搜索|查找|grep|search)\s+[`"\']?([^`"\'\n,，。]+)[`"\']?', re.IGNORECASE),
        # "在 xxx 中搜索 yyy"
        re.compile(
            r'在\s*[`"\']?([a-zA-Z0-9_\-./\\*]+)[`"\']?\s*中(?:搜索|查找)\s*[`"\']?([^`"\'\n]+)[`"\']?', re.IGNORECASE),
        # grep pattern 或 grep "pattern"
        re.compile(
            r'grep\s+(?:-[a-zA-Z]+\s+)*[`"\']?([^`"\'\s]+)[`"\']?', re.IGNORECASE),
    ]

    # Glob 文件匹配模式 - 更宽松的匹配
    GLOB_PATTERNS = [
        # "查找文件 *.py" 或 "列出所有 *.js"
        re.compile(
            r'(?:查找|列出|find|list)\s*(?:所有|all)?\s*(?:文件|files?)?\s*[`"\']?([a-zA-Z0-9_\-./\\*?]+\*[a-zA-Z0-9_\-./\\*?]*)[`"\']?', re.IGNORECASE),
        # 直接匹配 *.py 或 **/*.ts 模式
        re.compile(r'[`"\'](\*+\.?\w+)[`"\']', re.IGNORECASE),
        # 匹配 "所有 .py 文件"
        re.compile(
            r'(?:所有|all)\s*[`"\']?\.?(\w+)[`"\']?\s*(?:文件|files?)', re.IGNORECASE),
    ]

    # WebSearch 网络搜索模式
    WEBSEARCH_PATTERNS = [
        # "搜索网络" 或 "在线搜索"
        re.compile(
            r'(?:网络搜索|在线搜索|web\s*search|google|百度)[:\s]+[`"\']?([^`"\']+)[`"\']?', re.IGNORECASE),
    ]

    # WebFetch 获取网页模式
    WEBFETCH_PATTERNS = [
        # "获取网页 xxx" 或 "访问 url"
        re.compile(
            r'(?:获取|访问|fetch|get)\s*(?:网页|url|链接)?[:\s]*[`"\']?(https?://[^\s`"\']+)[`"\']?', re.IGNORECASE),
    ]

    # TodoWrite 待办事项模式
    TODO_PATTERNS = [
        # "TODO: xxx" 或 "待办: xxx"
        re.compile(r'(?:TODO|待办|任务)[:\s]+([^\n]+)', re.IGNORECASE),
        # "添加待办 xxx"
        re.compile(r'(?:添加|创建)\s*(?:待办|任务|todo)[:\s]+([^\n]+)', re.IGNORECASE),
    ]

    # KillShell 终止进程模式
    KILLSHELL_PATTERNS = [
        # "终止进程" 或 "停止运行"
        re.compile(r'(?:终止|停止|kill|stop)\s*(?:进程|shell|运行|process)',
                   re.IGNORECASE),
    ]

    # 语言到文件扩展名映射
    LANG_TO_EXT = {
        'python': '.py',
        'py': '.py',
        'javascript': '.js',
        'js': '.js',
        'typescript': '.ts',
        'ts': '.ts',
        'tsx': '.tsx',
        'jsx': '.jsx',
        'html': '.html',
        'css': '.css',
        'scss': '.scss',
        'sass': '.sass',
        'json': '.json',
        'yaml': '.yaml',
        'yml': '.yml',
        'toml': '.toml',
        'xml': '.xml',
        'sql': '.sql',
        'sh': '.sh',
        'bash': '.sh',
        'shell': '.sh',
        'powershell': '.ps1',
        'ps1': '.ps1',
        'rust': '.rs',
        'go': '.go',
        'java': '.java',
        'kotlin': '.kt',
        'swift': '.swift',
        'c': '.c',
        'cpp': '.cpp',
        'c++': '.cpp',
        'csharp': '.cs',
        'cs': '.cs',
        'ruby': '.rb',
        'php': '.php',
        'lua': '.lua',
        'r': '.R',
        'scala': '.scala',
        'vue': '.vue',
        'svelte': '.svelte',
        'markdown': '.md',
        'md': '.md',
    }

    def __init__(self, available_tools: Optional[List[Dict[str, Any]]] = None):
        """初始化工具模拟器

        Args:
            available_tools: Claude Code 传递的可用工具列表
        """
        self.available_tools = available_tools or []
        self.tool_names = {t.get('name') for t in self.available_tools}
        logger.debug(f"[ToolSimulator] 可用工具: {self.tool_names}")

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否可用"""
        return tool_name in self.tool_names or not self.available_tools

    def extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """从文本中提取代码块

        Args:
            text: Grok 响应文本

        Returns:
            代码块列表
        """
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1) or ''
            content = match.group(2).strip()

            if content:  # 忽略空代码块
                blocks.append(CodeBlock(
                    language=language.lower(),
                    content=content,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        logger.debug(f"[ToolSimulator] 提取到 {len(blocks)} 个代码块")
        return blocks

    def _is_valid_file_path(self, path: str) -> bool:
        """验证文件路径是否有效（排除 URL 域名等）"""
        if not path:
            return False

        path_lower = path.lower()

        # 排除明显的 URL 域名和云服务相关
        url_patterns = [
            '.com', '.cn', '.org', '.net', '.io', '.dev', '.app',
            '.gov', '.edu', '.co', '.info', '.biz',
            'http:', 'https:', 'www.',
            # 云服务相关
            'cos.ap', 'oss.', 's3.', 'blob.', 'storage.',
            'myqcloud', 'aliyun', 'amazonaws',
            # 常见域名模式
            '.ap-', '.eu-', '.us-', '.cn-',
        ]
        for pattern in url_patterns:
            if pattern in path_lower:
                logger.debug(f"[ToolSimulator] 排除 URL/云服务路径: {path}")
                return False

        # 排除只有扩展名没有文件名的情况（如 ".py"）
        if path.startswith('.') and '/' not in path and '\\' not in path:
            logger.debug(f"[ToolSimulator] 排除无效路径: {path}")
            return False

        # 必须有合理的扩展名
        if '.' not in path:
            logger.debug(f"[ToolSimulator] 排除无扩展名路径: {path}")
            return False

        # 文件名必须至少有 2 个字符（不含扩展名）
        filename = path.split('/')[-1].split('\\')[-1]
        name_part = filename.rsplit('.', 1)[0]
        if len(name_part) < 2:
            logger.debug(f"[ToolSimulator] 排除文件名太短: {path}")
            return False

        # 扩展名必须是常见的代码/配置文件扩展名
        valid_extensions = {
            'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'scss', 'sass',
            'json', 'yaml', 'yml', 'toml', 'xml', 'sql', 'sh', 'bash',
            'rs', 'go', 'java', 'kt', 'swift', 'c', 'cpp', 'h', 'hpp',
            'cs', 'rb', 'php', 'lua', 'r', 'scala', 'vue', 'svelte',
            'md', 'txt', 'cfg', 'ini', 'env', 'gitignore', 'dockerfile',
            'makefile', 'gradle', 'pom', 'lock', 'sum',
        }
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in valid_extensions:
            logger.debug(f"[ToolSimulator] 排除非代码扩展名: {path} (ext={ext})")
            return False

        return True

    def infer_file_path(self, text: str, code_block: CodeBlock, context_before: str = '') -> Optional[str]:
        """推断代码块的目标文件路径

        Args:
            text: 完整响应文本
            code_block: 代码块
            context_before: 代码块之前的上下文

        Returns:
            推断的文件路径，如果无法推断则返回 None
        """
        # 1. 优先检查代码块内容的第一行是否是文件路径注释（如 # app.py）
        first_line = code_block.content.split('\n')[0].strip()
        if first_line.startswith('#') or first_line.startswith('//') or first_line.startswith('<!--'):
            path_match = re.search(
                r'[#/<>!\-]+\s*([a-zA-Z0-9_\-./\\]+\.\w+)', first_line)
            if path_match:
                file_path = path_match.group(1).replace('\\', '/')
                if self._is_valid_file_path(file_path):
                    logger.debug(f"[ToolSimulator] 从代码注释推断文件路径: {file_path}")
                    return file_path

        # 2. 在代码块紧邻的上下文中查找 Markdown 格式的文件名
        search_text = context_before if context_before else text[:code_block.start_pos]
        # 只搜索最近的 200 字符（稍微扩大范围以捕获 **filename** 格式）
        search_text = search_text[-200:] if len(
            search_text) > 200 else search_text

        # 2a. 优先查找 **filename.ext** 格式（Markdown 加粗）
        md_bold_pattern = re.compile(r'\*\*([a-zA-Z0-9_\-./\\]+\.\w+)\*\*')
        md_matches = list(md_bold_pattern.finditer(search_text))
        if md_matches:
            file_path = md_matches[-1].group(1).replace('\\', '/')
            if self._is_valid_file_path(file_path):
                logger.debug(
                    f"[ToolSimulator] 从 Markdown 加粗推断文件路径: {file_path}")
                return file_path

        # 2b. 查找 ### filename.ext 或 ### 1. filename.ext 格式（带可选数字前缀）
        # 同时处理带括号说明的情况，如 ### 1. app.py（说明）
        md_header_pattern = re.compile(
            r'#{1,6}\s*(?:\d+\.\s*)?([a-zA-Z0-9_\-./\\]+\.\w+)')
        header_matches = list(md_header_pattern.finditer(search_text))
        if header_matches:
            file_path = header_matches[-1].group(1).replace('\\', '/')
            if self._is_valid_file_path(file_path):
                logger.debug(
                    f"[ToolSimulator] 从 Markdown 标题推断文件路径: {file_path}")
                return file_path

        # 2c. 查找 `filename.ext` 格式（代码引用）
        backtick_pattern = re.compile(r'`([a-zA-Z0-9_\-./\\]+\.\w+)`')
        backtick_matches = list(backtick_pattern.finditer(search_text))
        if backtick_matches:
            file_path = backtick_matches[-1].group(1).replace('\\', '/')
            if self._is_valid_file_path(file_path):
                logger.debug(f"[ToolSimulator] 从代码引用推断文件路径: {file_path}")
                return file_path

        # 2d. 查找 "1. filename.ext" 或 "2. path/to/file.ext" 格式（编号列表）
        numbered_pattern = re.compile(
            r'\d+\.\s*([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)*\.\w+)')
        numbered_matches = list(numbered_pattern.finditer(search_text))
        if numbered_matches:
            file_path = numbered_matches[-1].group(1).replace('\\', '/')
            if self._is_valid_file_path(file_path):
                logger.debug(f"[ToolSimulator] 从编号列表推断文件路径: {file_path}")
                return file_path

        # 3. 查找传统的文件路径模式
        last_match = None
        for pattern in self.FILE_PATH_PATTERNS:
            for match in pattern.finditer(search_text):
                candidate = match.group(1).replace('\\', '/')
                if self._is_valid_file_path(candidate):
                    last_match = candidate

        if last_match:
            logger.debug(f"[ToolSimulator] 从上下文推断文件路径: {last_match}")
            return last_match

        # 4. 不再使用默认文件名（避免错误匹配）
        # 如果无法确定文件路径，不生成 Write 调用
        return None

    def detect_bash_command(self, text: str) -> Optional[str]:
        """检测文本中的 Bash 命令意图

        Args:
            text: 响应文本

        Returns:
            检测到的命令，如果没有则返回 None
        """
        for pattern in self.BASH_PATTERNS:
            match = pattern.search(text)
            if match:
                command = match.group(1).strip()
                if command:
                    logger.debug(f"[ToolSimulator] 检测到 Bash 命令: {command}")
                    return command
        return None

    def generate_tool_id(self) -> str:
        """生成工具调用 ID"""
        return f"toolu_{uuid.uuid4().hex[:24]}"

    def create_write_tool_call(self, file_path: str, content: str) -> ToolCall:
        """创建 Write 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Write",
            input={
                "file_path": file_path,
                "content": content
            }
        )

    def create_bash_tool_call(self, command: str) -> ToolCall:
        """创建 Bash 工具调用"""
        # 自动转换 Windows 命令为 bash 等效命令
        converted_command = self._convert_windows_to_bash(command)
        return ToolCall(
            id=self.generate_tool_id(),
            name="Bash",
            input={
                "command": converted_command
            }
        )

    def _convert_windows_to_bash(self, command: str) -> str:
        """将 Windows CMD 命令转换为 bash 等效命令

        支持的转换：
        - del /Q "path" → rm -f "path"
        - del "path" → rm "path"
        - copy → cp
        - move → mv
        - type → cat
        - dir → ls
        - mkdir → mkdir -p
        - rmdir /S /Q → rm -rf
        - cls → clear
        - echo. → echo
        """
        original_command = command
        converted = command.strip()

        # Windows del 命令转换为 rm
        # del /Q "path" → rm -f "path"
        # del "path" → rm "path"
        del_pattern = re.compile(
            r'^del\s+(?:/[QqFf]\s+)*["\']?(.+?)["\']?\s*$',
            re.IGNORECASE
        )
        del_match = del_pattern.match(converted)
        if del_match:
            path = del_match.group(1).strip().strip('"\'')
            # 如果原命令有 /Q 或 /F 参数，使用 rm -f
            if re.search(r'/[QqFf]', converted, re.IGNORECASE):
                converted = f'rm -f "{path}"'
            else:
                converted = f'rm "{path}"'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows copy 命令转换为 cp
        copy_pattern = re.compile(
            r'^copy\s+(?:/[YyBbAa]\s+)*["\']?(.+?)["\']?\s+["\']?(.+?)["\']?\s*$',
            re.IGNORECASE
        )
        copy_match = copy_pattern.match(converted)
        if copy_match:
            src = copy_match.group(1).strip().strip('"\'')
            dst = copy_match.group(2).strip().strip('"\'')
            converted = f'cp "{src}" "{dst}"'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows move 命令转换为 mv
        move_pattern = re.compile(
            r'^move\s+(?:/[Yy]\s+)*["\']?(.+?)["\']?\s+["\']?(.+?)["\']?\s*$',
            re.IGNORECASE
        )
        move_match = move_pattern.match(converted)
        if move_match:
            src = move_match.group(1).strip().strip('"\'')
            dst = move_match.group(2).strip().strip('"\'')
            converted = f'mv "{src}" "{dst}"'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows type 命令转换为 cat
        type_pattern = re.compile(
            r'^type\s+["\']?(.+?)["\']?\s*$',
            re.IGNORECASE
        )
        type_match = type_pattern.match(converted)
        if type_match:
            path = type_match.group(1).strip().strip('"\'')
            converted = f'cat "{path}"'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows dir 命令转换为 ls
        dir_pattern = re.compile(
            r'^dir\s*(?:/[BbSsWwAaDdOoNnPpQqRrTtXxLl4]\s*)*(["\']?.+?["\']?)?\s*$',
            re.IGNORECASE
        )
        dir_match = dir_pattern.match(converted)
        if dir_match:
            path = dir_match.group(1)
            if path:
                path = path.strip().strip('"\'')
                converted = f'ls -la "{path}"'
            else:
                converted = 'ls -la'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows rmdir /S /Q 命令转换为 rm -rf
        rmdir_pattern = re.compile(
            r'^rmdir\s+(?:/[SsQq]\s+)+["\']?(.+?)["\']?\s*$',
            re.IGNORECASE
        )
        rmdir_match = rmdir_pattern.match(converted)
        if rmdir_match:
            path = rmdir_match.group(1).strip().strip('"\'')
            converted = f'rm -rf "{path}"'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows cls 命令转换为 clear
        if converted.lower() == 'cls':
            converted = 'clear'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # Windows echo. (空行) 转换为 echo
        if converted.lower() == 'echo.':
            converted = 'echo'
            logger.info(
                f"[ToolSimulator] 命令转换: {original_command} → {converted}")
            return converted

        # 将 Windows 路径分隔符 \ 转换为 /
        if '\\' in converted and not converted.startswith('\\\\'):
            # 排除网络路径 \\server\share
            converted = converted.replace('\\', '/')
            if converted != original_command:
                logger.info(
                    f"[ToolSimulator] 路径分隔符转换: {original_command} → {converted}")

        return converted

    def create_read_tool_call(self, file_path: str) -> ToolCall:
        """创建 Read 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Read",
            input={
                "file_path": file_path
            }
        )

    def create_grep_tool_call(self, pattern: str, path: str = ".") -> ToolCall:
        """创建 Grep 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Grep",
            input={
                "pattern": pattern,
                "path": path
            }
        )

    def create_glob_tool_call(self, pattern: str) -> ToolCall:
        """创建 Glob 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Glob",
            input={
                "pattern": pattern
            }
        )

    def create_websearch_tool_call(self, query: str) -> ToolCall:
        """创建 WebSearch 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="WebSearch",
            input={
                "query": query
            }
        )

    def create_webfetch_tool_call(self, url: str) -> ToolCall:
        """创建 WebFetch 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="WebFetch",
            input={
                "url": url
            }
        )

    def create_todowrite_tool_call(self, content: str, status: str = "pending") -> ToolCall:
        """创建 TodoWrite 工具调用

        Claude Code 期望的格式：
        {
          "todos": [
            {"id": "unique-id", "content": "内容", "status": "pending", "activeForm": "checkbox"}
          ]
        }
        """
        todo_id = f"todo_{uuid.uuid4().hex[:8]}"
        return ToolCall(
            id=self.generate_tool_id(),
            name="TodoWrite",
            input={
                "todos": [
                    {
                        "id": todo_id,
                        "content": content,
                        "status": status,
                        "activeForm": "checkbox"  # Claude Code 必需字段
                    }
                ]
            }
        )

    def create_killshell_tool_call(self) -> ToolCall:
        """创建 KillShell 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="KillShell",
            input={}
        )

    def create_task_tool_call(self, description: str) -> ToolCall:
        """创建 Task 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Task",
            input={
                "description": description
            }
        )

    def create_notebookedit_tool_call(self, notebook_path: str, cell_index: int, content: str) -> ToolCall:
        """创建 NotebookEdit 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="NotebookEdit",
            input={
                "notebook_path": notebook_path,
                "cell_index": cell_index,
                "content": content
            }
        )

    def create_edit_tool_call(self, file_path: str, old_string: str, new_string: str) -> ToolCall:
        """创建 Edit 工具调用"""
        return ToolCall(
            id=self.generate_tool_id(),
            name="Edit",
            input={
                "file_path": file_path,
                "old_string": old_string,
                "new_string": new_string
            }
        )

    def detect_read_intent(self, text: str, user_message: str = '') -> Optional[str]:
        """检测读取文件的意图

        Args:
            text: 响应文本
            user_message: 用户消息（用于上下文）

        Returns:
            文件路径，如果没有检测到则返回 None
        """
        # 先从响应文本中检测
        for pattern in self.READ_PATTERNS:
            match = pattern.search(text)
            if match:
                file_path = match.group(1).replace('\\', '/')
                logger.debug(f"[ToolSimulator] 从响应检测到 Read 意图: {file_path}")
                return file_path

        # 再从用户消息中检测（如果用户请求读取文件）
        if user_message:
            for pattern in self.READ_PATTERNS:
                match = pattern.search(user_message)
                if match:
                    file_path = match.group(1).replace('\\', '/')
                    logger.debug(
                        f"[ToolSimulator] 从用户消息检测到 Read 意图: {file_path}")
                    return file_path

        return None

    def detect_edit_intent(self, text: str, code_blocks: List[CodeBlock]) -> Optional[Tuple[str, str, str]]:
        """检测编辑文件的意图

        Args:
            text: 响应文本
            code_blocks: 代码块列表

        Returns:
            (file_path, old_string, new_string) 或 None
        """
        # 查找 diff 格式的代码块
        for block in code_blocks:
            if block.language == 'diff':
                # 尝试解析 diff 内容
                lines = block.content.split('\n')
                old_lines = []
                new_lines = []
                file_path = None

                for line in lines:
                    if line.startswith('---'):
                        # 尝试从 --- 行提取文件路径
                        path_match = re.search(r'---\s+(?:a/)?([^\s]+)', line)
                        if path_match:
                            file_path = path_match.group(1)
                    elif line.startswith('-') and not line.startswith('---'):
                        old_lines.append(line[1:])
                    elif line.startswith('+') and not line.startswith('+++'):
                        new_lines.append(line[1:])

                if file_path and (old_lines or new_lines):
                    old_string = '\n'.join(old_lines)
                    new_string = '\n'.join(new_lines)
                    logger.debug(
                        f"[ToolSimulator] 检测到 Edit 意图 (diff): {file_path}")
                    return (file_path, old_string, new_string)

        # 查找明确的编辑模式
        for pattern in self.EDIT_PATTERNS:
            match = pattern.search(text)
            if match:
                file_path = match.group(1).replace('\\', '/')
                # 查找相邻的代码块作为新内容
                for block in code_blocks:
                    if block.content.strip():
                        # 简化处理：假设整个代码块是新内容
                        logger.debug(
                            f"[ToolSimulator] 检测到 Edit 意图: {file_path}")
                        return (file_path, "", block.content)

        return None

    def detect_grep_intent(self, text: str, code_blocks: List[CodeBlock], user_message: str = '') -> Optional[Tuple[str, str]]:
        """检测 Grep 搜索意图

        Returns:
            (pattern, path) 或 None
        """
        # 先从文本模式匹配
        for pattern in self.GREP_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    return (groups[0], groups[1])
                elif len(groups) >= 1:
                    return (groups[0], ".")

        # 从用户消息中检测（如果只有 Grep 工具可用）
        if user_message and len(self.tool_names) <= 2 and 'Grep' in self.tool_names:
            for pattern in self.GREP_PATTERNS:
                match = pattern.search(user_message)
                if match:
                    groups = match.groups()
                    if len(groups) >= 1:
                        logger.debug(
                            f"[ToolSimulator] 从用户消息检测到 Grep 意图: {groups[0]}")
                        return (groups[0], ".")

        # 从 shell 代码块中提取 grep 命令
        for block in code_blocks:
            if block.language in ('sh', 'bash', 'shell', 'powershell', 'cmd'):
                # 匹配 grep pattern path 或 grep "pattern" path
                grep_match = re.search(
                    r'grep\s+(?:-[a-zA-Z]+\s+)*["\']?([^"\']+)["\']?\s+([^\s|>]+)', block.content)
                if grep_match:
                    return (grep_match.group(1), grep_match.group(2))
                # 简单的 grep pattern
                grep_simple = re.search(
                    r'grep\s+(?:-[a-zA-Z]+\s+)*["\']?([^"\'|\s>]+)["\']?', block.content)
                if grep_simple:
                    return (grep_simple.group(1), ".")

        return None

    def _is_valid_glob_pattern(self, pattern: str) -> bool:
        """验证是否是有效的 glob 模式（必须包含 * 或 ?）"""
        if not pattern:
            return False
        # 必须包含 glob 通配符
        if '*' not in pattern and '?' not in pattern:
            return False
        # 排除太短的模式
        if len(pattern) < 2:
            return False
        return True

    def detect_glob_intent(self, text: str, code_blocks: List[CodeBlock]) -> Optional[str]:
        """检测 Glob 文件匹配意图"""
        # 先从文本模式匹配
        for pattern in self.GLOB_PATTERNS:
            match = pattern.search(text)
            if match:
                result = match.group(1)
                if self._is_valid_glob_pattern(result):
                    return result

        # 从 shell 代码块中提取 ls/find 命令中的 glob 模式
        for block in code_blocks:
            if block.language in ('sh', 'bash', 'shell', 'powershell', 'cmd'):
                # 匹配 ls *.py 或 find . -name "*.py"
                ls_match = re.search(
                    r'(?:ls|dir)\s+([*?][^\s|>]*|\S*[*?]\S*)', block.content)
                if ls_match:
                    result = ls_match.group(1)
                    if self._is_valid_glob_pattern(result):
                        return result
                find_match = re.search(
                    r'find\s+\S+\s+-name\s+["\']?([*?][^"\'|\s]*)["\']?', block.content)
                if find_match:
                    result = find_match.group(1)
                    if self._is_valid_glob_pattern(result):
                        return result

        # 检测文本中的通配符模式
        wildcard_match = re.search(r'[`"\'](\*\.[a-zA-Z]+)[`"\']', text)
        if wildcard_match:
            result = wildcard_match.group(1)
            if self._is_valid_glob_pattern(result):
                return result

        return None

    def detect_websearch_intent(self, text: str) -> Optional[str]:
        """检测 WebSearch 网络搜索意图"""
        for pattern in self.WEBSEARCH_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def detect_webfetch_intent(self, text: str) -> Optional[str]:
        """检测 WebFetch 获取网页意图"""
        for pattern in self.WEBFETCH_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def detect_todo_intent(self, text: str) -> List[str]:
        """检测 TodoWrite 待办事项意图

        注意：从普通文本中检测 TODO 很容易误匹配，因此使用严格的规则：
        - 必须是明确的 TODO 列表格式（如 "- [ ] 任务"）
        - 内容必须是合理的任务描述（5-100 字符）
        """
        todos = []
        # 只匹配明确的复选框格式：- [ ] 或 * [ ]
        checkbox_pattern = re.compile(r'[-*]\s*\[\s*\]\s*([^\n]{5,100})')
        for match in checkbox_pattern.finditer(text):
            content = match.group(1).strip()
            # 排除看起来像是代码或 URL 的内容
            if not any(x in content for x in ['http', 'www.', '```', '://', '()']):
                todos.append(content)
        return todos

    def detect_killshell_intent(self, text: str) -> bool:
        """检测 KillShell 终止进程意图"""
        for pattern in self.KILLSHELL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def extract_bash_from_blocks(self, code_blocks: List[CodeBlock]) -> List[str]:
        """从代码块中提取 Bash 命令

        Args:
            code_blocks: 代码块列表

        Returns:
            Bash 命令列表
        """
        commands = []
        for block in code_blocks:
            if block.language in ('sh', 'bash', 'shell', 'powershell', 'ps1', 'cmd', 'zsh'):
                # 过滤掉注释和空行
                lines = []
                for line in block.content.split('\n'):
                    line = line.strip()
                    # 跳过空行和纯注释行
                    if not line or line.startswith('#'):
                        continue
                    # 跳过 Windows CMD 注释
                    if line.startswith('::'):
                        continue
                    # 跳过输出示例（通常以特定模式开头）
                    if line.startswith('$') or line.startswith('>') or line.startswith('>>>'):
                        line = line.lstrip('$> ')
                    lines.append(line)

                if lines:
                    # 合并多行命令或逐行执行
                    command = ' && '.join(lines) if len(
                        lines) > 1 else lines[0]
                    # 过滤掉太长或可能危险的命令
                    if len(command) < 500:  # 限制命令长度
                        commands.append(command)
                        logger.debug(
                            f"[ToolSimulator] 提取 Bash 命令: {command[:50]}...")

        return commands

    def _is_hallucinated_content(self, content: str, file_path: str) -> bool:
        """检测内容是否是模型幻觉（说明文字而非真正的文件内容）

        幻觉内容的特征：
        1. 包含完成状态消息（"已成功"、"100%"、"完成"等）
        2. 包含对用户的说明/提示（"你需要"、"请访问"、"现在可以"等）
        3. 包含项目结构说明（"← 这是..."等箭头标注）
        4. 内容与文件类型明显不匹配

        Args:
            content: 文件内容
            file_path: 文件路径

        Returns:
            True 如果检测到幻觉内容，False 否则
        """
        if not content or not content.strip():
            return True

        content_lower = content.lower()
        content_lines = content.strip().split('\n')
        first_line = content_lines[0].strip() if content_lines else ""

        # 检测幻觉模式
        hallucination_patterns = [
            # 完成状态消息
            r'已[成功完]*(?:安装|启动|创建|配置|部署|运行)',
            r'100\s*%\s*(?:成功|完成|安装)',
            r'项目(?:已|完全)?(?:可运行|可开发|可部署)',
            r'全部(?:完成|成功)',

            # 对用户的说明
            r'访问\s*https?://',
            r'打开\s*https?://',
            r'请(?:访问|打开|运行|执行)',
            r'你(?:现在)?(?:可以|需要|应该)',
            r'现在(?:可以|你可以)',

            # 箭头标注说明
            r'←\s*(?:这是|你|只需|由|可以)',
            r'→\s*(?:这是|你|只需|由|可以)',

            # 项目结构/文件说明
            r'(?:你)?只需要维护',
            r'自动生成并提交',
            r'完全可以删掉',
            r'没人用了',
            r'过时了',

            # 模型自我说明
            r'我(?:已经|将要|会|正在)',
            r'让我(?:为你|帮你)',
            r'接下来(?:我会|将)',
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(
                    f"[ToolSimulator] 检测到幻觉内容 (匹配: {pattern}): {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

        # 检测文件类型与内容不匹配
        file_ext = file_path.rsplit(
            '.', 1)[-1].lower() if '.' in file_path else ''
        file_name = file_path.rsplit('/', 1)[-1].rsplit('\\', 1)[-1].lower()

        # pyproject.toml 必须是完整有效的配置文件
        if file_name == 'pyproject.toml':
            # 必须包含 [project] 或 [build-system] 表
            has_required_tables = (
                '[project]' in content or
                '[build-system]' in content
            )
            if not has_required_tables:
                logger.warning(
                    f"[ToolSimulator] 检测到不完整的 pyproject.toml: {file_path}")
                logger.warning(
                    f"[ToolSimulator] 缺少 [project] 或 [build-system] 表")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

            # 内容长度检查（完整的 pyproject.toml 至少有 15 行）
            if len(content_lines) < 15:
                logger.warning(
                    f"[ToolSimulator] pyproject.toml 内容太短 ({len(content_lines)} 行): {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

            # 检测是否以注释+裸键值对开头（常见的错误模式）
            if first_line.startswith('#') and '修复' in first_line:
                logger.warning(
                    f"[ToolSimulator] 检测到错误的 pyproject.toml 内容（以修复注释开头）: {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True
        elif file_ext == 'toml':
            # 其他 TOML 文件：检查是否是有效的 TOML（应该包含 [ 或 key = value）
            if not re.search(r'^\s*\[|\s*\w+\s*=', content, re.MULTILINE):
                logger.warning(f"[ToolSimulator] 检测到无效 TOML 内容: {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

        # .lock 文件通常有特定格式（JSON、TOML 或特定结构）
        if file_ext == 'lock':
            # uv.lock 应该是 TOML 格式，以 version = 开头
            # package-lock.json 应该以 { 开头
            if not (content.strip().startswith('{') or
                    content.strip().startswith('version') or
                    content.strip().startswith('[')):
                logger.warning(f"[ToolSimulator] 检测到无效 lock 文件内容: {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

        # JSON 文件应该以 { 或 [ 开头
        if file_ext == 'json':
            stripped = content.strip()
            if not (stripped.startswith('{') or stripped.startswith('[')):
                logger.warning(f"[ToolSimulator] 检测到无效 JSON 内容: {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

        # Python 文件不应该只有中文说明
        if file_ext == 'py':
            # 检查是否有任何 Python 语法元素
            has_python_syntax = any([
                'import ' in content,
                'from ' in content,
                'def ' in content,
                'class ' in content,
                'if ' in content,
                'for ' in content,
                'while ' in content,
                '=' in content,
                'print(' in content,
                '#' in content and any(c.isascii() for c in content),
            ])
            if not has_python_syntax:
                logger.warning(f"[ToolSimulator] 检测到无效 Python 内容: {file_path}")
                logger.warning(f"[ToolSimulator] 内容预览: {content[:200]}...")
                return True

        return False

    def _is_valid_bash_command(self, cmd: str) -> bool:
        """检查 Bash 命令是否适合自动执行

        过滤掉：
        1. Windows CMD 语法（cd /d, ::, copy）
        2. URL 被误解析为命令
        3. 包含明显占位符的命令
        4. 可能危险的命令

        Args:
            cmd: Bash 命令

        Returns:
            True 如果命令可以执行，False 否则
        """
        # 过滤掉空命令
        if not cmd or len(cmd.strip()) < 2:
            return False

        # 过滤掉以 URL 开头的"命令"
        if cmd.startswith('http://') or cmd.startswith('https://'):
            logger.debug(f"[ToolSimulator] 过滤 URL 命令: {cmd[:50]}...")
            return False

        # 过滤掉 Windows CMD 语法
        windows_patterns = [
            'cd /d ',  # Windows cd 语法
            ':: ',     # Windows 注释
            'copy ',   # Windows copy (用空格区分)
            'xcopy ',  # Windows xcopy
            'notepad',  # Windows 记事本
        ]
        cmd_lower = cmd.lower()
        for pattern in windows_patterns:
            if pattern in cmd_lower:
                logger.debug(f"[ToolSimulator] 过滤 Windows 命令: {cmd[:50]}...")
                return False

        # 过滤掉包含明显占位符的命令
        placeholder_patterns = [
            'your-username',
            'your_username',
            'your-',
            '你的',
            '<your',
            '[your',
            'example.com',
            'yourdomain',
        ]
        for pattern in placeholder_patterns:
            if pattern.lower() in cmd_lower:
                logger.debug(f"[ToolSimulator] 过滤占位符命令: {cmd[:50]}...")
                return False

        # 过滤掉可能危险的命令
        dangerous_patterns = [
            'rm -rf /',
            'rm -rf ~',
            'rm -rf *',
            'sudo rm -rf',
            ':(){:|:&};:',  # fork 炸弹
        ]
        for pattern in dangerous_patterns:
            if pattern in cmd:
                logger.debug(f"[ToolSimulator] 过滤危险命令: {cmd[:50]}...")
                return False

        return True

    def _extract_balanced_json(self, text: str, start_pos: int) -> Optional[Tuple[int, int]]:
        """提取平衡的 JSON 对象（处理嵌套的大括号）

        Args:
            text: 完整文本
            start_pos: JSON 开始位置（第一个 { 的位置）

        Returns:
            (start, end) 位置元组，如果失败返回 None
        """
        if start_pos >= len(text) or text[start_pos] != '{':
            return None

        depth = 0
        in_string = False
        escape_next = False
        i = start_pos

        while i < len(text):
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                i += 1
                continue

            if in_string:
                i += 1
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return (start_pos, i + 1)

            i += 1

        return None  # 未找到匹配的结束括号

    def _parse_grok_tool_calls(self, text: str) -> List[ToolCall]:
        """解析 Grok 模拟的工具调用格式

        Grok 有时会返回类似这样的格式：
        [Tool Call: Write]
        {"file_path":"hello.py","content":"print('hello')"}
        [/Tool Call]

        改进：使用平衡括号匹配来正确处理多行和嵌套 JSON

        Args:
            text: Grok 响应文本

        Returns:
            解析出的工具调用列表
        """
        tool_calls = []
        parsed_positions = set()  # 避免重复解析同一位置

        # 先尝试标准格式（带代码块）
        for alt_match in self.GROK_TOOL_CALL_PATTERN_ALT.finditer(text):
            start_pos = alt_match.end()
            tool_name = alt_match.group(1)

            # 查找结束标记
            end_match = self.TOOL_CALL_END_PATTERN.search(text, start_pos)
            if not end_match:
                continue

            json_start = start_pos
            json_end = end_match.start()
            json_content = text[json_start:json_end].strip()

            # 移除代码块标记
            json_content = re.sub(r'^```(?:json)?\s*', '', json_content)
            json_content = re.sub(r'\s*```$', '', json_content)
            json_content = json_content.strip()

            # 查找 JSON 对象的实际开始位置
            json_obj_start = json_content.find('{')
            if json_obj_start == -1:
                continue

            json_obj_start += json_start + \
                json_content[:json_obj_start].count('\n')

            # 使用平衡括号匹配提取完整 JSON
            json_range = self._extract_balanced_json(text, json_obj_start)
            if json_range:
                json_start, json_end = json_range
                json_content = text[json_start:json_end]

                pos_key = (alt_match.start(), end_match.end())
                if pos_key in parsed_positions:
                    continue

                try:
                    tool_input = orjson.loads(json_content)

                    # Bash 工具：自动转换 Windows 命令为 bash 等效命令
                    if tool_name == "Bash" and "command" in tool_input:
                        original_cmd = tool_input["command"]
                        converted_cmd = self._convert_windows_to_bash(
                            original_cmd)
                        if converted_cmd != original_cmd:
                            tool_input["command"] = converted_cmd
                            logger.info(
                                f"[ToolSimulator] Bash 命令已转换 (带代码块): {original_cmd} → {converted_cmd}")

                    if self.has_tool(tool_name):
                        tool_call = ToolCall(
                            id=self.generate_tool_id(),
                            name=tool_name,
                            input=tool_input
                        )
                        tool_calls.append(tool_call)
                        parsed_positions.add(pos_key)
                        logger.info(
                            f"[ToolSimulator] 解析 Grok 工具调用 (带代码块): {tool_name}")
                except Exception as e:
                    logger.warning(
                        f"[ToolSimulator] 解析 Grok 工具调用失败 ({tool_name}, 带代码块): {e}")
                    continue

        # 再尝试标准格式（不带代码块）
        for match in self.GROK_TOOL_CALL_PATTERN.finditer(text):
            start_pos = match.end()
            tool_name = match.group(1)

            # 查找结束标记
            end_match = self.TOOL_CALL_END_PATTERN.search(text, start_pos)
            if not end_match:
                continue

            json_start = start_pos
            json_end = end_match.start()
            json_content = text[json_start:json_end].strip()

            # 查找 JSON 对象的实际开始位置
            json_obj_start = json_content.find('{')
            if json_obj_start == -1:
                continue

            json_obj_start += json_start

            # 使用平衡括号匹配提取完整 JSON
            json_range = self._extract_balanced_json(text, json_obj_start)
            if json_range:
                json_start, json_end = json_range
                json_content = text[json_start:json_end]

                pos_key = (match.start(), end_match.end())
                if pos_key in parsed_positions:
                    continue

                try:
                    tool_input = orjson.loads(json_content)

                    # 根据官方文档，Skill 工具的参数是 command（保持不变）
                    # 但 Claude Code 可能期望 skill，所以同时保留两个参数以确保兼容性
                    if tool_name == "Skill" and "command" in tool_input:
                        # 保留 command，同时添加 skill 以确保兼容性
                        tool_input["skill"] = tool_input["command"]
                        logger.debug(
                            "[ToolSimulator] Skill 工具同时设置 command 和 skill 参数以确保兼容性")

                    # Bash 工具：自动转换 Windows 命令为 bash 等效命令
                    if tool_name == "Bash" and "command" in tool_input:
                        original_cmd = tool_input["command"]
                        converted_cmd = self._convert_windows_to_bash(
                            original_cmd)
                        if converted_cmd != original_cmd:
                            tool_input["command"] = converted_cmd
                            logger.info(
                                f"[ToolSimulator] Bash 命令已转换: {original_cmd} → {converted_cmd}")

                    # 检查工具是否可用
                    if self.has_tool(tool_name):
                        tool_call = ToolCall(
                            id=self.generate_tool_id(),
                            name=tool_name,
                            input=tool_input
                        )
                        tool_calls.append(tool_call)
                        parsed_positions.add(pos_key)
                        logger.info(
                            f"[ToolSimulator] 解析 Grok 工具调用: {tool_name}")
                except Exception as e:
                    logger.warning(
                        f"[ToolSimulator] 解析 Grok 工具调用失败 ({tool_name}): {e}")
                    continue

        return tool_calls

    def _clean_tool_results(self, text: str) -> str:
        """清理 Grok 返回的工具执行结果，防止循环

        只移除:
        - [Tool Result]...[/Tool Result] 格式（之前执行的结果回显）

        保留:
        - [Tool Call: ...]...[/Tool Call] 格式（这是我们要解析的）
        """
        original_len = len(text)

        # 只移除 [Tool Result]...[/Tool Result] 格式
        cleaned = re.sub(
            r'\[Tool Result\].*?\[/Tool Result\]',
            '',
            text,
            flags=re.DOTALL
        )

        if len(cleaned) < original_len:
            logger.debug(
                f"[ToolSimulator] 清理了工具结果，移除 {original_len - len(cleaned)} 字符")

        return cleaned.strip()

    def _remove_tool_call_tags(self, text: str) -> str:
        """从文本中移除已解析的 [Tool Call: ...] 格式

        在工具调用成功解析后调用，避免文本内容重复显示工具调用

        改进：使用更精确的匹配，避免误删正常内容
        """
        # 移除 [Tool Call: ...]...[/Tool Call] 格式
        # 使用更精确的匹配：确保匹配完整的工具调用块
        cleaned = re.sub(
            r'\[Tool Call:\s*\w+\]\s*\n?.*?\[/Tool Call\]',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # 清理多余的空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def parse_response(self, text: str, user_message: str = '') -> Tuple[str, List[ToolCall]]:
        """解析 Grok 响应，提取工具调用

        Args:
            text: Grok 响应文本
            user_message: 用户消息（用于上下文推断）

        Returns:
            (处理后的文本, 工具调用列表)
        """
        tool_calls = []

        # 清理之前的工具执行结果（防止循环），但保留 [Tool Call: ...] 格式
        remaining_text = self._clean_tool_results(text)

        # 检查是否有可用的工具
        if not self.available_tools:
            logger.debug("[ToolSimulator] 没有可用工具，跳过工具模拟")
            return remaining_text, []

        # 0. 优先解析 Grok 的 [Tool Call: ...] 格式（通过系统提示词规范化的格式）
        # 这是最可靠的方式，因为格式是明确的
        grok_tool_calls = self._parse_grok_tool_calls(remaining_text)
        if grok_tool_calls:
            logger.info(
                f"[ToolSimulator] 从 Grok 格式提取到 {len(grok_tool_calls)} 个工具调用")
            # 过滤重复的文件路径，只保留每个文件的第一个调用
            seen_files = set()
            seen_commands = set()
            valid_calls = []
            for call in grok_tool_calls:
                if call.name == "Write":
                    file_path = call.input.get("file_path", "")
                    content = call.input.get("content", "")
                    # 跳过重复文件、空内容、太短的内容（可能是示例输出）
                    if file_path in seen_files:
                        logger.debug(f"[ToolSimulator] 跳过重复文件: {file_path}")
                        continue
                    if len(content.strip()) < 10:
                        logger.debug(
                            f"[ToolSimulator] 跳过内容太短的 Write: {file_path} ({len(content)} chars)")
                        continue
                    # 检测幻觉内容（说明文字而非真正的文件内容）
                    if self._is_hallucinated_content(content, file_path):
                        logger.warning(
                            f"[ToolSimulator] 跳过幻觉内容的 Write: {file_path}")
                        continue
                    seen_files.add(file_path)
                    valid_calls.append(call)
                    logger.info(f"[ToolSimulator] 保留 Write 工具调用: {file_path}")
                elif call.name == "Edit":
                    file_path = call.input.get("file_path", "")
                    new_string = call.input.get("new_string", "")
                    if file_path in seen_files:
                        logger.debug(
                            f"[ToolSimulator] 跳过重复文件的 Edit: {file_path}")
                        continue
                    # 检测幻觉内容（只检查 new_string，因为 old_string 可能是空的）
                    if new_string and self._is_hallucinated_content(new_string, file_path):
                        logger.warning(
                            f"[ToolSimulator] 跳过幻觉内容的 Edit: {file_path}")
                        continue
                    seen_files.add(file_path)
                    valid_calls.append(call)
                    logger.info(f"[ToolSimulator] 保留 Edit 工具调用: {file_path}")
                elif call.name == "Bash":
                    command = call.input.get("command", "")
                    # 规范化命令用于去重比较（移除多余空格，统一引号）
                    normalized_cmd = ' '.join(command.split())  # 合并多余空格
                    normalized_cmd = normalized_cmd.replace('"', "'")  # 统一使用单引号
                    if normalized_cmd in seen_commands:
                        logger.debug(
                            f"[ToolSimulator] 跳过重复命令: {command[:30]}...")
                        continue
                    seen_commands.add(normalized_cmd)
                    valid_calls.append(call)
                    logger.info(
                        f"[ToolSimulator] 保留 Bash 工具调用: {command[:50]}...")
                elif call.name == "Read":
                    file_path = call.input.get("file_path", "")
                    if file_path in seen_files:
                        logger.debug(
                            f"[ToolSimulator] 跳过重复文件的 Read: {file_path}")
                        continue
                    seen_files.add(file_path)
                    valid_calls.append(call)
                    logger.info(f"[ToolSimulator] 保留 Read 工具调用: {file_path}")
                else:
                    # 其他工具直接保留
                    valid_calls.append(call)
                    logger.info(f"[ToolSimulator] 保留 {call.name} 工具调用")

            if valid_calls:
                # 防止死循环：如果同时有文件操作(Edit/Write)和Bash命令，只保留文件操作
                # 这样可以确保文件先被修改，Bash 命令在下一轮对话中执行
                has_file_ops = any(c.name in ('Edit', 'Write')
                                   for c in valid_calls)
                has_bash = any(c.name == 'Bash' for c in valid_calls)

                if has_file_ops and has_bash:
                    # 移除 Bash 调用，只保留文件操作
                    original_count = len(valid_calls)
                    valid_calls = [c for c in valid_calls if c.name != 'Bash']
                    removed_count = original_count - len(valid_calls)
                    logger.warning(
                        f"[ToolSimulator] 防止死循环：移除了 {removed_count} 个 Bash 调用，"
                        f"因为同时有文件操作。文件操作必须先完成。"
                    )

                logger.info(f"[ToolSimulator] 有效工具调用数: {len(valid_calls)}")
                # 从文本中移除已解析的 [Tool Call] 格式，避免重复
                cleaned_text = self._remove_tool_call_tags(remaining_text)
                return cleaned_text, valid_calls
            # 如果所有调用都被过滤了，继续尝试从代码块提取

        # 1. 提取代码块（使用清理后的文本）
        code_blocks = self.extract_code_blocks(remaining_text)

        # 跟踪已处理的文件路径，避免重复
        processed_files = set()
        processed_commands = set()

        # 2. 检测 Read 意图
        if self.has_tool('Read'):
            read_file = self.detect_read_intent(remaining_text, user_message)
            if read_file and read_file not in processed_files:
                read_call = self.create_read_tool_call(read_file)
                tool_calls.append(read_call)
                processed_files.add(read_file)
                logger.info(f"[ToolSimulator] 创建 Read 工具调用: {read_file}")
            # 如果只有 Read 工具可用，尝试从代码块推断文件并创建 Read 调用
            elif len(self.tool_names) == 1 and 'Read' in self.tool_names:
                for block in code_blocks:
                    context_before = remaining_text[:block.start_pos]
                    file_path = self.infer_file_path(
                        remaining_text, block, context_before)
                    if file_path and file_path not in processed_files:
                        read_call = self.create_read_tool_call(file_path)
                        tool_calls.append(read_call)
                        processed_files.add(file_path)
                        logger.info(
                            f"[ToolSimulator] 创建 Read 工具调用（从代码块）: {file_path}")
                        break  # 只创建一个 Read 调用

        # 3. 检测 Edit 意图
        if self.has_tool('Edit'):
            edit_result = self.detect_edit_intent(remaining_text, code_blocks)
            if edit_result:
                file_path, old_str, new_str = edit_result
                if file_path not in processed_files:
                    edit_call = self.create_edit_tool_call(
                        file_path, old_str, new_str)
                    tool_calls.append(edit_call)
                    processed_files.add(file_path)
                    logger.info(f"[ToolSimulator] 创建 Edit 工具调用: {file_path}")

        # 4. 处理 Bash 命令（从 shell 代码块提取，但需要严格过滤）
        if self.has_tool('Bash'):
            bash_commands = self.extract_bash_from_blocks(code_blocks)
            for cmd in bash_commands:
                # 过滤掉不适合自动执行的命令
                if self._is_valid_bash_command(cmd) and cmd not in processed_commands:
                    bash_call = self.create_bash_tool_call(cmd)
                    tool_calls.append(bash_call)
                    processed_commands.add(cmd)
                    logger.info(f"[ToolSimulator] 创建 Bash 工具调用: {cmd[:50]}...")

        # 5. 检测 Grep 搜索意图
        if self.has_tool('Grep'):
            grep_result = self.detect_grep_intent(
                remaining_text, code_blocks, user_message)
            if grep_result:
                pattern, path = grep_result
                grep_call = self.create_grep_tool_call(pattern, path)
                tool_calls.append(grep_call)
                logger.info(
                    f"[ToolSimulator] 创建 Grep 工具调用: {pattern} in {path}")

        # 6-9. 以下工具的自动检测已禁用（从纯文本猜测容易误匹配）
        # 这些工具现在依赖 Grok 使用 [Tool Call: ...] 格式明确调用
        # 通过系统提示词注入，Grok 会知道如何使用正确格式
        # - Glob: 文件匹配
        # - WebSearch: 网络搜索
        # - WebFetch: 获取网页
        # - TodoWrite: 待办事项
        # 如果 Grok 返回了 [Tool Call: Glob/WebSearch/WebFetch/TodoWrite]，
        # 会在上面的 _parse_grok_tool_calls 中被正确解析

        # 10. 检测 KillShell 终止进程意图
        if self.has_tool('KillShell'):
            if self.detect_killshell_intent(remaining_text):
                kill_call = self.create_killshell_tool_call()
                tool_calls.append(kill_call)
                logger.info(f"[ToolSimulator] 创建 KillShell 工具调用")

        # 5. 处理 Write 或 Edit（从非 shell 代码块提取）
        for block in code_blocks:
            # shell 代码块已经在上面处理为 Bash
            if block.language in ('sh', 'bash', 'shell', 'powershell', 'ps1', 'cmd', 'zsh'):
                continue

            # 跳过 diff 代码块（已经在 Edit 中处理）
            if block.language == 'diff':
                continue

            # 跳过太短的代码块（可能是输出示例）
            if len(block.content.strip()) < 10:
                logger.debug(
                    f"[ToolSimulator] 跳过太短的代码块: {block.content[:30]}...")
                continue

            # 推断文件路径
            context_before = remaining_text[:block.start_pos]
            file_path = self.infer_file_path(
                remaining_text, block, context_before)

            if file_path and file_path not in processed_files:
                # 检测幻觉内容（说明文字而非真正的文件内容）
                if self._is_hallucinated_content(block.content, file_path):
                    logger.warning(f"[ToolSimulator] 跳过幻觉内容的代码块: {file_path}")
                    continue

                # 优先使用 Write 工具
                if self.has_tool('Write'):
                    write_call = self.create_write_tool_call(
                        file_path, block.content)
                    tool_calls.append(write_call)
                    processed_files.add(file_path)
                    logger.info(f"[ToolSimulator] 创建 Write 工具调用: {file_path}")
                # 如果没有 Write 但有 Edit，使用 Edit（空 old_string 表示创建/覆盖）
                elif self.has_tool('Edit'):
                    edit_call = self.create_edit_tool_call(
                        file_path, "", block.content)
                    tool_calls.append(edit_call)
                    processed_files.add(file_path)
                    logger.info(
                        f"[ToolSimulator] 创建 Edit 工具调用（覆盖）: {file_path}")

        logger.info(f"[ToolSimulator] 共生成 {len(tool_calls)} 个工具调用")

        # Claude Code 安全机制要求：Write/Edit 必须在同一轮响应中先有 Read
        # 自动在 Write/Edit 调用之前插入 Read 调用
        if tool_calls and self.has_tool('Read'):
            final_calls = []
            read_files = set()  # 已添加 Read 的文件

            for tc in tool_calls:
                if tc.name in ('Write', 'Edit'):
                    file_path = tc.input.get('file_path', '')
                    if file_path and file_path not in read_files:
                        # 在 Write/Edit 之前插入 Read 调用
                        read_call = self.create_read_tool_call(file_path)
                        final_calls.append(read_call)
                        read_files.add(file_path)
                        logger.info(
                            f"[ToolSimulator] 自动插入 Read 调用（满足 Claude Code 安全要求）: {file_path}")
                final_calls.append(tc)

            tool_calls = final_calls
            logger.info(f"[ToolSimulator] 添加 Read 后工具调用数: {len(tool_calls)}")

        return remaining_text, tool_calls

    def to_anthropic_content(self, text: str, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """将解析结果转换为 Anthropic content 格式

        Args:
            text: 处理后的文本
            tool_calls: 工具调用列表

        Returns:
            Anthropic content 块列表
        """
        content = []

        # 添加文本内容（如果有）
        if text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        # 添加工具调用
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input
            })

        return content

    def process_response(self, text: str, user_message: str = '') -> List[Dict[str, Any]]:
        """处理 Grok 响应，返回 Anthropic 格式的 content

        Args:
            text: Grok 响应文本
            user_message: 用户消息（用于上下文）

        Returns:
            Anthropic content 块列表
        """
        remaining_text, tool_calls = self.parse_response(text, user_message)
        return self.to_anthropic_content(remaining_text, tool_calls)
