"""Anthropic 请求-响应模型定义"""

from fastapi import HTTPException
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator

from app.models.grok_models import Models


class AnthropicMessage(BaseModel):
    """Anthropic 消息格式"""
    role: str = Field(..., description="角色: user 或 assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="消息内容")


class AnthropicChatRequest(BaseModel):
    """Anthropic 聊天请求"""
    
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
    # Claude Code 可能发送的额外字段
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具列表")
    tool_choice: Optional[Dict[str, Any]] = Field(None, description="工具选择")

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
        """验证模型名称 - 支持 Anthropic 模型名映射"""
        # Anthropic 模型名映射到 Grok 模型
        anthropic_to_grok = {
            "claude-3-5-sonnet-20241022": "grok-2-latest",
            "claude-3-5-sonnet-latest": "grok-2-latest",
            "claude-3-opus-20240229": "grok-2-latest",
            "claude-3-opus-latest": "grok-2-latest",
            "claude-3-sonnet-20240229": "grok-2-1212",
            "claude-3-haiku-20240307": "grok-2-1212",
            "claude-2.1": "grok-2",
            "claude-2.0": "grok-2",
        }
        
        # 如果是 Anthropic 模型名，映射到 Grok 模型
        if v in anthropic_to_grok:
            return anthropic_to_grok[v]
        
        # 否则直接验证是否是有效的 Grok 模型
        if not Models.is_valid_model(v):
            supported = Models.get_all_model_names()
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型 '{v}', 支持的模型: {', '.join(supported)}"
            )
        return v


class AnthropicContentBlock(BaseModel):
    """Anthropic 内容块"""
    type: str = Field(..., description="类型")
    text: Optional[str] = Field(None, description="文本内容")


class AnthropicUsage(BaseModel):
    """Anthropic token 使用统计"""
    input_tokens: int = Field(..., description="输入token数")
    output_tokens: int = Field(..., description="输出token数")


class AnthropicChatResponse(BaseModel):
    """Anthropic 聊天响应"""
    id: str = Field(..., description="响应ID")
    type: str = Field("message", description="类型")
    role: str = Field("assistant", description="角色")
    model: str = Field(..., description="模型")
    content: List[AnthropicContentBlock] = Field(..., description="内容块")
    stop_reason: Optional[str] = Field(None, description="停止原因")
    stop_sequence: Optional[str] = Field(None, description="停止序列")
    usage: AnthropicUsage = Field(..., description="使用统计")


class AnthropicStreamEvent(BaseModel):
    """Anthropic 流式事件"""
    type: str = Field(..., description="事件类型")
    index: Optional[int] = Field(None, description="索引")
    delta: Optional[Dict[str, Any]] = Field(None, description="增量数据")
    content_block: Optional[Dict[str, Any]] = Field(None, description="内容块")
    message: Optional[Dict[str, Any]] = Field(None, description="消息")
    usage: Optional[AnthropicUsage] = Field(None, description="使用统计")


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic token 计数请求"""
    messages: List[AnthropicMessage] = Field(..., description="消息列表")
    model: str = Field(..., description="模型名称")
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="系统提示词")


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic token 计数响应"""
    input_tokens: int = Field(..., description="输入token数")







