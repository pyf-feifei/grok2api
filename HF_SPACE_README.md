---
title: Grok2API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Grok2API

基于 FastAPI 的 Grok API 转换服务，支持流式对话、图像生成、视频生成、联网搜索、深度思考等功能。

## 部署说明

此版本已针对 Hugging Face Space 进行优化。

### 环境变量配置

请在 Space Settings -> Repository secrets 中配置以下变量：

| 变量名 | 必填 | 说明 | 默认值 |
|--------|------|------|--------|
| `ADMIN_USERNAME` | 否 | 管理后台用户名 | admin |
| `ADMIN_PASSWORD` | **是** | 管理后台密码 | admin |
| `API_KEY` | 否 | API 访问密钥 | 空 |
| `BASE_URL` | **是** | 服务基础 URL | 空 |
| `PROXY_URL` | 否 | HTTP 代理地址 | 空 |
| `STORAGE_MODE` | 否 | 存储模式 (file/mysql/redis) | file |
| `DATABASE_URL` | 否 | 数据库连接 URL | 空 |

### 重要提示

1. **BASE_URL 设置**：请设置为你的 Space URL，例如 `https://your-username-grok2api.hf.space`

2. **持久化存储**：
   - 免费版 Space 重启后数据会丢失
   - 建议使用外部 MySQL/Redis 进行持久化
   - 设置 `STORAGE_MODE=mysql` 或 `STORAGE_MODE=redis`
   - 配置对应的 `DATABASE_URL`

3. **安全性**：
   - 请务必修改默认的 `ADMIN_PASSWORD`
   - 建议设置 `API_KEY` 限制 API 访问

## 使用方法

1. 访问 Space URL 登录管理后台
2. 在管理后台添加 Grok Token
3. 使用 OpenAI 兼容格式调用 API

### API 端点

- `POST /v1/chat/completions` - 聊天对话
- `GET /v1/models` - 获取模型列表
- `POST /v1/audio/speech` - 文本转语音

## 注意事项

- 本项目仅供学习与研究使用
- 请遵守 Grok 和 Hugging Face 的使用条款
