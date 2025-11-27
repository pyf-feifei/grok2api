# Anthropic API 快速开始

## 🎉 新增功能

Grok2API 现已支持 **Anthropic Claude API** 兼容接口！

## 📦 新增文件

```
app/
├── models/
│   └── anthropic_schema.py          # Anthropic 请求/响应模型
├── api/
│   └── v1/
│       └── anthropic.py             # Anthropic API 路由
└── services/
    └── anthropic/
        ├── __init__.py
        └── converter.py             # 格式转换器

test_anthropic.py                    # Python 测试脚本
test_anthropic.sh                    # Bash 测试脚本
ANTHROPIC_API.md                     # 完整使用文档
ANTHROPIC_QUICKSTART.md              # 本文件
```

## 🚀 快速测试

### 1. 使用 Python SDK

```bash
# 安装 SDK
pip install anthropic

# 设置环境变量
export GROK2API_API_KEY="your-api-key"
export GROK2API_BASE_URL="http://localhost:9527/v1"

# 运行测试
python test_anthropic.py
```

### 2. 使用 curl

```bash
# 设置环境变量
export GROK2API_API_KEY="your-api-key"
export GROK2API_BASE_URL="http://localhost:9527"

# 运行测试
chmod +x test_anthropic.sh
./test_anthropic.sh
```

### 3. 直接测试

```bash
curl -X POST http://localhost:9527/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Hello! 介绍一下你自己。"
      }
    ]
  }'
```

## 🔧 核心特性

### ✅ 已支持

- **文本对话** - 单轮和多轮对话
- **系统提示词** - 独立的 system 参数
- **流式响应** - Server-Sent Events
- **多模态** - 图片输入支持
- **参数控制** - temperature, top_p, max_tokens
- **模型映射** - Claude 模型名自动映射到 Grok

### 📋 支持的模型映射

| Anthropic 模型 | 映射到 Grok 模型 |
|----------------|------------------|
| `claude-3-5-sonnet-20241022` | `grok-2-latest` |
| `claude-3-5-sonnet-latest` | `grok-2-latest` |
| `claude-3-opus-20240229` | `grok-2-latest` |
| `claude-3-sonnet-20240229` | `grok-2-1212` |
| `claude-3-haiku-20240307` | `grok-2-1212` |

## 🔌 API 端点

```
POST /v1/messages
```

与 Anthropic 官方 API 兼容，支持所有标准参数。

## 📖 详细文档

查看 [ANTHROPIC_API.md](./ANTHROPIC_API.md) 获取完整使用文档，包括：

- 详细的参数说明
- 多种语言示例（Python, JavaScript, curl）
- 流式响应处理
- 多模态使用
- 错误处理
- 最佳实践

## 🔄 与现有接口对比

| 特性 | OpenAI 接口 | Anthropic 接口 |
|------|-------------|----------------|
| 端点 | `/v1/chat/completions` | `/v1/messages` |
| 系统消息 | 在 messages 中 | 独立 `system` 参数 |
| 流式格式 | SSE (data) | SSE (event types) |
| 支持角色 | system/user/assistant | user/assistant |
| SDK | openai | anthropic |

## 🎯 使用场景

1. **现有 Claude 用户迁移** - 无需修改代码，只需更换 base_url
2. **多接口支持** - 同时支持 OpenAI 和 Anthropic 格式
3. **SDK 兼容** - 使用官方 SDK 直接接入
4. **统一后端** - 一个服务支持多种格式

## ⚠️ 注意事项

1. **认证方式** - 使用 Grok2API 的认证密钥，而非 Anthropic 密钥
2. **模型限制** - 实际使用 Grok 模型，受 Grok 配额限制
3. **功能支持** - 某些 Claude 特有功能可能不完全支持
4. **响应格式** - 完全兼容 Anthropic API 规范

## 💡 提示

- 首次使用建议先运行测试脚本验证功能
- 查看日志文件 `logs/app.log` 排查问题
- 使用流式响应获得更好的用户体验
- 合理设置 max_tokens 控制成本

## 🆘 获取帮助

- 📖 查看完整文档: [ANTHROPIC_API.md](./ANTHROPIC_API.md)
- 🐛 遇到问题: 检查日志文件或提交 Issue
- 💬 社区讨论: GitHub Discussions

## 🎊 开始使用

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="your-grok2api-key",
    base_url="http://localhost:9527/v1"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

**就是这么简单！** 🚀



