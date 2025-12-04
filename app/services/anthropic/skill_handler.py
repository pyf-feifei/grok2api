"""Skill 工具处理器 - 处理 Claude Code 的 Skill 工具调用"""

import os
import json
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.core.logger import logger


class SkillHandler:
    """处理 Skill 工具调用，读取本地技能目录"""

    # 默认技能目录（Windows）
    DEFAULT_SKILLS_DIR = Path.home() / ".claude" / "skills"

    @classmethod
    def get_skills_directory(cls) -> Path:
        """获取技能目录路径"""
        # 支持环境变量覆盖
        skills_dir = os.getenv("CLAUDE_SKILLS_DIR")
        if skills_dir:
            return Path(skills_dir)
        return cls.DEFAULT_SKILLS_DIR

    @classmethod
    def list_skills(cls) -> List[Dict[str, Any]]:
        """列出所有可用的技能

        Returns:
            技能列表，每个技能包含 name, description, version 等信息
        """
        skills_dir = cls.get_skills_directory()
        skills = []

        if not skills_dir.exists():
            logger.warning(f"[SkillHandler] 技能目录不存在: {skills_dir}")
            return skills

        logger.info(f"[SkillHandler] 扫描技能目录: {skills_dir}")

        # 查找所有 JSON 文件（技能元数据文件）
        json_files = list(skills_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    skill_data = json.load(f)

                # 提取关键信息
                skill_info = {
                    "name": skill_data.get("name", json_file.stem),
                    "description": skill_data.get("description", ""),
                    "version": skill_data.get("version", "1.0.0"),
                    "author": skill_data.get("author", ""),
                    "category": skill_data.get("category", ""),
                    "tags": skill_data.get("tags", []),
                    "capabilities": skill_data.get("capabilities", []),
                    "file": str(json_file.name),
                }

                # 检查是否有对应的 MD 文件
                md_file = json_file.with_suffix('.md')
                if md_file.exists():
                    skill_info["has_documentation"] = True
                    skill_info["documentation_file"] = md_file.name
                else:
                    skill_info["has_documentation"] = False

                skills.append(skill_info)
                logger.debug(f"[SkillHandler] 找到技能: {skill_info['name']}")

            except Exception as e:
                logger.warning(f"[SkillHandler] 解析技能文件失败 {json_file}: {e}")
                continue

        logger.info(f"[SkillHandler] 共找到 {len(skills)} 个技能")
        return skills

    @classmethod
    def format_skills_response(cls, skills: List[Dict[str, Any]]) -> str:
        """格式化技能列表为文本响应

        Args:
            skills: 技能列表

        Returns:
            格式化的文本响应
        """
        if not skills:
            return "您当前没有安装任何技能。\n\n技能存储在 `~/.claude/skills` 目录下，每个技能包含一个 JSON 元数据文件和一个可选的 MD 文档文件。"

        lines = [f"您当前安装了 {len(skills)} 个技能：\n"]

        for i, skill in enumerate(skills, 1):
            lines.append(f"\n## {i}. {skill['name']}")
            lines.append(f"**版本**: {skill.get('version', '1.0.0')}")

            if skill.get('description'):
                lines.append(f"**描述**: {skill['description']}")

            if skill.get('author'):
                lines.append(f"**作者**: {skill['author']}")

            if skill.get('category'):
                lines.append(f"**分类**: {skill['category']}")

            if skill.get('tags'):
                tags_str = ", ".join(skill['tags'])
                lines.append(f"**标签**: {tags_str}")

            if skill.get('capabilities'):
                lines.append(f"**功能**:")
                for cap in skill['capabilities']:
                    lines.append(f"  - {cap}")

            if skill.get('has_documentation'):
                lines.append(
                    f"**文档**: 已包含 ({skill.get('documentation_file', '')})")

        return "\n".join(lines)

    @classmethod
    def load_skill_prompt(cls, skill_name: str) -> Optional[str]:
        """加载指定技能的 SKILL.md 文件内容

        Args:
            skill_name: 技能名称

        Returns:
            SKILL.md 文件内容，如果不存在则返回 None
        """
        skills_dir = cls.get_skills_directory()

        # 尝试多种可能的文件名
        possible_files = [
            skills_dir / f"{skill_name}.md",
            skills_dir / skill_name / "SKILL.md",
            skills_dir / skill_name / "skill.md",
        ]

        for md_file in possible_files:
            if md_file.exists():
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info(f"[SkillHandler] 加载技能提示: {md_file}")
                    return content
                except Exception as e:
                    logger.warning(f"[SkillHandler] 读取技能文件失败 {md_file}: {e}")
                    continue

        logger.warning(f"[SkillHandler] 未找到技能文件: {skill_name}")
        return None

    @classmethod
    def handle_skill_tool_call(cls, tool_input: Dict[str, Any]) -> str:
        """处理 Skill 工具调用

        Args:
            tool_input: 工具输入参数，包含 command 字段（技能名称）

        Returns:
            技能列表的文本响应（如果 command 为空）或技能提示内容
        """
        logger.info(f"[SkillHandler] 处理 Skill 工具调用: {tool_input}")

        # Claude Code 期望 Skill 工具的参数是 skill（兼容 command）
        skill_name = tool_input.get(
            "skill", tool_input.get("command", "")).strip()
        if skill_name:
            skill_prompt = cls.load_skill_prompt(skill_name)
            if skill_prompt:
                return skill_prompt
            else:
                return f"未找到技能 '{skill_name}' 的提示文件。请检查技能是否正确安装。"

        # 如果没有 command，返回技能列表
        try:
            skills = cls.list_skills()
            response = cls.format_skills_response(skills)
            return response
        except Exception as e:
            logger.error(f"[SkillHandler] 处理失败: {e}")
            return f"获取技能列表时出错: {str(e)}"
