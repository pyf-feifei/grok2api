"""Anthropic 请求-响应模型定义 - 完全兼容 Claude Code"""

from fastapi import HTTPException
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

from app.models.grok_models import Models


class AnthropicMessage(BaseModel):
    """Anthropic 消息格式"""
    # 允许额外字段，以兼容 Claude Code 发送的各种内容类型
    model_config = ConfigDict(extra='allow')
    
    role: str = Field(..., description="角色: user 或 assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="消息内容")


class AnthropicChatRequest(BaseModel):
    """Anthropic 聊天请求 - 完全兼容 Claude Code
    
    参考: https://api-docs.deepseek.com/guides/anthropic_api
    支持 Claude Code 发送的所有字段，未知字段会被忽略
    """
    # 允许额外字段，以兼容 Claude Code 发送的各种参数
    model_config = ConfigDict(extra='allow')
    
    model: str = Field(..., description="模型名称", min_length=1)
    messages: List[AnthropicMessage] = Field(..., description="消息列表", min_length=1)
    max_tokens: int = Field(4096, ge=1, le=100000, description="最大Token数")
    # system 可以是字符串或数组（Claude Code 发送数组格式）
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="系统提示词")
    temperature: Optional[float] = Field(1.0, ge=0, le=2, description="采样温度")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="采样参数")
    top_k: Optional[int] = Field(None, ge=1, description="Top-K采样")
    stream: Optional[bool] = Field(False, description="流式响应")
    stop_sequences: Optional[List[str]] = Field(None, description="停止序列")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    
    # Claude Code 工具相关字段
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具列表")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="工具选择")
    
    # Claude Code 扩展思考相关字段（interleaved-thinking beta）
    thinking: Optional[Dict[str, Any]] = Field(None, description="扩展思考配置，包含 type 和 budget_tokens")
    
    # Claude Code 其他可能字段
    container: Optional[Dict[str, Any]] = Field(None, description="容器配置（忽略）")
    mcp_servers: Optional[List[Dict[str, Any]]] = Field(None, description="MCP服务器配置（忽略）")
    service_tier: Optional[str] = Field(None, description="服务层级（忽略）")
    betas: Optional[List[str]] = Field(None, description="Beta 功能列表（忽略）")

    @classmethod
    @field_validator('messages')
    def validate_messages(cls, v):
        """验证消息格式"""
        if not v:
            raise HTTPException(status_code=400, detail="messages: field required")
        
        for msg in v:
            if msg.role not in ['user', 'assistant']:
                raise HTTPException(
                    status_code=400,
                    detail=f"messages.{v.index(msg)}.role: Input should be 'user' or 'assistant'"
                )
        
        return v

    @classmethod
    @field_validator('model')
    def validate_model(cls, v):
        """验证模型名称 - 支持 Anthropic 模型名映射
        
        参考 DeepSeek 实现：接受任何模型名，未知模型自动映射到默认模型
        """
        # Anthropic 模型名映射到 Grok 模型（扩展映射表）
        anthropic_to_grok = {
            # Claude 3.5 系列
            "claude-3-5-sonnet-20241022": "grok-4.1",
            "claude-3-5-sonnet-latest": "grok-4.1",
            "claude-3-5-haiku-20241022": "grok-4-fast",
            "claude-3-5-haiku-latest": "grok-4-fast",
            # Claude 3 系列
            "claude-3-opus-20240229": "grok-4.1",
            "claude-3-opus-latest": "grok-4.1",
            "claude-3-sonnet-20240229": "grok-4.1",
            "claude-3-haiku-20240307": "grok-4-fast",
            # Claude 4 系列（新版本）
            "claude-sonnet-4-5-20250929": "grok-4.1",
            "claude-sonnet-4-20250514": "grok-4.1",
            "claude-opus-4-0-20250514": "grok-4.1",
            # Claude Haiku 4.5（Claude Code 使用的）
            "claude-haiku-4-5-20251001": "grok-4-fast",
            # Claude 2 系列
            "claude-2.1": "grok-4.1",
            "claude-2.0": "grok-4.1",
        }
        
        # 如果是 Anthropic 模型名，映射到 Grok 模型
        if v in anthropic_to_grok:
            return anthropic_to_grok[v]
        
        # 如果是有效的 Grok 模型，直接返回
        if Models.is_valid_model(v):
            return v
        
        # 对于未知模型，不报错，而是使用默认模型（参考 DeepSeek 实现）
        # 这样可以兼容 Claude Code 发送的任何模型名
        from app.core.logger import logger
        logger.warning(f"[Anthropic] 未知模型 '{v}'，将使用默认模型 grok-4.1")
        return "grok-4.1"


class AnthropicUsage(BaseModel):
    """Anthropic token 使用统计"""
    input_tokens: int = Field(..., description="输入token数")
    output_tokens: int = Field(..., description="输出token数")


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic token 计数请求 - 完全兼容 Claude Code"""
    # 允许额外字段
    model_config = ConfigDict(extra='allow')
    
    messages: List[AnthropicMessage] = Field(..., description="消息列表")
    model: str = Field(..., description="模型名称")
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="系统提示词")
    # Claude Code 可能发送的额外字段
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具列表")
    thinking: Optional[Dict[str, Any]] = Field(None, description="扩展思考配置")


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic token 计数响应"""
    input_tokens: int = Field(..., description="输入token数")



