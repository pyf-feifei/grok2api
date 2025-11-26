# 快速开始 - LiteLLM + Cline 配置指南

## 系统架构

```
Cline (VS Code) → LiteLLM (代理层) → Grok2API (你的服务) → Grok API
    端口: -           端口: 4000          端口: 8001          端口: -
```

## 前提条件

- ✅ Python 3.8+ 已安装
- ✅ Grok2API 服务已运行（`python main.py`，默认端口 8001）
- ✅ Cline 扩展已安装在 VS Code

## 启动 LiteLLM

LiteLLM 作为一个中间层，负责标准化 Function Calling 请求并转发给 Grok2API。

### 1. 启动服务

使用提供的 Python 脚本一键启动（会自动检查配置）：

```bash
# 方式一：使用 uv (推荐)
uv run python start_litellm.py

# 方式二：直接使用 python
python start_litellm.py
```

启动成功后，你应该看到类似以下的日志：
```
INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
```

**注意**：请保持这个终端窗口打开！

### 2. 配置 Cline

项目已包含 `.vscode/settings.json`，VS Code 应会自动读取。如果需要手动配置：

1. 打开 Cline 设置 (`Cmd/Ctrl + Shift + P` -> `Cline: Configure`)
2. 选择 **OpenAI Compatible**
3. 填写配置：
   - **Base URL**: `http://localhost:4000`
   - **API Key**: `sk-test` (任意值，LiteLLM 会自动使用 Grok2API 配置的 key)
   - **Model**: `grok-model` (或配置文件中列出的其他模型，如 `grok-4.1`)

## 常见问题

### LiteLLM 启动失败？
- 检查端口 4000 是否被占用
- 确保已安装 litellm: `pip install litellm` 或 `uv sync`

### Cline 无法连接？
- 检查 LiteLLM 是否正在运行
- 检查 Base URL 是否正确 (不要包含 `/v1/chat/completions`)

### API Key 如何配置？
- LiteLLM 会自动读取 `data/setting.toml` 中的 `api_key`
- 无需在 Cline 中重复配置真实的 Key
