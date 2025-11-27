# Anthropic API 接口说明

## 📝 概述

Grok2API 现已支持 **Anthropic Claude API** 兼容接口！

**端点**: `POST http://localhost:8002/v1/messages`

---

## 🚀 快速开始

### 1. 安装 SDK

```bash
pip install anthropic
```

### 2. 使用示例

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="your-grok2api-key",
    base_url="http://localhost:8002/v1"  # ← 注意端口 8002
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(message.content[0].text)
```

### 3. 系统提示词

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="你是一个 Python 专家",  # 系统提示词
    messages=[
        {"role": "user", "content": "如何读取 JSON？"}
    ]
)
```

### 4. 流式响应

```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "讲个故事"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## 🔧 模型映射

| Anthropic 模型 | Grok 模型 |
|----------------|-----------|
| claude-3-5-sonnet-20241022 | grok-2-latest |
| claude-3-opus-20240229 | grok-2-latest |
| claude-3-haiku-20240307 | grok-2-1212 |
| claude-2.1 | grok-2 |

也可以直接使用 Grok 模型名。

---

## 📖 详细文档

- **完整 API 文档**: `ANTHROPIC_API.md`
- **快速开始**: `ANTHROPIC_QUICKSTART.md`
- **测试示例**: `ANTHROPIC_DEMO.py` 或 `test_anthropic.py`

---

## ⚙️ 配置

### 访问管理后台

```
URL: http://localhost:8002/login
默认账号: admin / admin
```

在管理后台：
1. 添加 Grok Token
2. 配置代理（如果 IP 被拦截）

---

## ✨ 核心特性

- ✅ 完整的 Anthropic API 支持
- ✅ 自动模型名映射
- ✅ 流式和非流式响应
- ✅ 系统提示词
- ✅ 多模态（图片）支持
- ✅ 100% SDK 兼容

---

## 🧪 测试

```bash
# 运行完整演示
python ANTHROPIC_DEMO.py

# 或运行测试脚本
python test_anthropic.py
```

---

## ❓ 常见问题

**Q: 返回 403 错误？**  
A: 需要在管理后台配置有效的 Grok Token 和代理

**Q: 端口是多少？**  
A: 服务运行在 **8002** 端口（避免与 sqlbot 的 8000-8001 冲突）

**Q: 如何使用流式响应？**  
A: 使用 SDK 的 `messages.stream()` 方法

---

**就是这么简单！** 🚀



