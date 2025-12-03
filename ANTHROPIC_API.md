# Anthropic API 兼容接口使用指南

## 概述

Grok2API 现在支持 Anthropic Claude API 兼容接口！您可以使用 Anthropic SDK 或直接调用 API 来与 Grok 模型交互。

## 端点

```
POST /v1/messages
```

## 支持的功能

- ✅ 文本对话
- ✅ 系统提示词
- ✅ 流式响应
- ✅ 多模态（图片）
- ✅ 温度控制
- ✅ Token 限制

## 模型映射

Anthropic 模型名会自动映射到相应的 Grok 模型：

| Anthropic 模型 | Grok 模型 |
|---|---|
| `claude-3-5-sonnet-20241022` | `grok-2-latest` |
| `claude-3-5-sonnet-latest` | `grok-2-latest` |
| `claude-3-opus-20240229` | `grok-2-latest` |
| `claude-3-opus-latest` | `grok-2-latest` |
| `claude-3-sonnet-20240229` | `grok-2-1212` |
| `claude-3-haiku-20240307` | `grok-2-1212` |
| `claude-2.1` | `grok-2` |
| `claude-2.0` | `grok-2` |

您也可以直接使用 Grok 模型名，如 `grok-2-latest`, `grok-2`, `grok-vision` 等。

## 使用示例

### Python (使用 Anthropic SDK)

```python
from anthropic import Anthropic

# 初始化客户端
client = Anthropic(
    api_key="your-api-key",  # 您的 Grok2API 密钥
    base_url="http://localhost:9527/v1"  # 您的 Grok2API 地址
)

# 发送消息
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(message.content[0].text)
```

### 流式响应

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="your-api-key",
    base_url="http://localhost:9527/v1"
)

# 使用流式响应
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 系统提示词

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a helpful AI assistant specializing in Python programming.",
    messages=[
        {"role": "user", "content": "How do I sort a list in Python?"}
    ]
)

print(message.content[0].text)
```

### 多模态（图片）

```python
import base64

# 读取图片并转换为 base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ],
        }
    ],
)

print(message.content[0].text)
```

### 使用 curl

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
        "content": "Hello, Claude!"
      }
    ]
  }'
```

### JavaScript/TypeScript

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:9527/v1'
});

const message = await client.messages.create({
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 1024,
  messages: [
    {
      role: 'user',
      content: 'Hello, Claude!'
    }
  ]
});

console.log(message.content[0].text);
```

## 请求参数

| 参数 | 类型 | 必需 | 描述 |
|---|---|---|---|
| `model` | string | 是 | 模型名称 |
| `messages` | array | 是 | 消息列表 |
| `max_tokens` | integer | 是 | 最大生成 token 数 |
| `system` | string | 否 | 系统提示词 |
| `temperature` | float | 否 | 采样温度 (0-2) |
| `top_p` | float | 否 | 核采样参数 |
| `stream` | boolean | 否 | 是否使用流式响应 |
| `stop_sequences` | array | 否 | 停止序列 |

## 响应格式

### 非流式响应

```json
{
  "id": "msg_01ABC123",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-5-sonnet-20241022",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you today?"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

### 流式响应

流式响应使用 Server-Sent Events (SSE) 格式，包含以下事件类型：

1. `message_start` - 消息开始
2. `content_block_start` - 内容块开始
3. `content_block_delta` - 内容增量
4. `content_block_stop` - 内容块结束
5. `message_delta` - 消息元数据更新
6. `message_stop` - 消息结束

## 错误处理

```python
from anthropic import Anthropic, APIError

client = Anthropic(
    api_key="your-api-key",
    base_url="http://localhost:9527/v1"
)

try:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
except APIError as e:
    print(f"API Error: {e.message}")
```

## 与 OpenAI 接口的区别

| 特性 | OpenAI | Anthropic |
|---|---|---|
| 端点 | `/v1/chat/completions` | `/v1/messages` |
| 系统提示词 | 在 messages 中 | 独立的 `system` 参数 |
| 流式格式 | SSE with data | SSE with event types |
| 角色 | system/user/assistant | user/assistant (system 独立) |

## 注意事项

1. **认证**：使用与 OpenAI 接口相同的认证方式（Bearer Token）
2. **模型映射**：Anthropic 模型名会自动映射到对应的 Grok 模型
3. **Token 限制**：实际 token 限制取决于底层 Grok 模型
4. **兼容性**：大部分 Anthropic SDK 功能都支持，但某些高级特性可能不完全一致

## 完整示例

```python
from anthropic import Anthropic

def chat_with_claude():
    client = Anthropic(
        api_key="your-api-key",
        base_url="http://localhost:9527/v1"
    )
    
    # 非流式对话
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "Explain quantum computing in simple terms."
            }
        ]
    )
    
    print("Response:", response.content[0].text)
    print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

if __name__ == "__main__":
    chat_with_claude()
```

## 技术支持

如遇到问题，请查看：
- 项目主 README
- 日志文件 `logs/app.log`
- GitHub Issues

## 更新日志

- **v1.3.1**: 新增 Anthropic API 兼容接口
  - 支持 `/v1/messages` 端点
  - 支持 Anthropic SDK
  - 自动模型名映射
  - 流式和非流式响应
  - 多模态支持








