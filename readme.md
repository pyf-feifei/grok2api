# Grok2API

基于 **FastAPI** 重构的 Grok2API，全面适配最新 Web 调用格式，支持流式对话、图像生成、图像编辑、联网搜索、深度思考，号池并发与自动负载均衡一体化。

<br>

## 使用说明

### 调用次数与配额

- **普通账号（Basic）**：免费使用 **80 次 / 20 小时**
- **Super 账号**：配额待定（作者未测）
- 系统自动负载均衡各账号调用次数，可在**管理页面**实时查看用量与状态

### 图像生成功能

- 在对话内容中输入如“给我画一个月亮”自动触发图片生成
- 每次以 **Markdown 格式返回两张图片**，共消耗 4 次额度
- **注意：Grok 的图片直链受 403 限制，系统自动缓存图片到本地。必须正确设置 `Base Url` 以确保图片能正常显示！**

### 视频生成功能

- 选择 `grok-imagine-0.9` 模型，传入图片和提示词即可（方式和 OpenAI 的图片分析调用格式一致）
- **支持单张或多张图片生成视频**（多张图片时，系统会自动为每张图片创建 post，并在 message 中拼接所有图片 URL）
- 返回格式为 `<video src="{full_video_url}" controls="controls"></video>`
- **注意：Grok 的视频直链受 403 限制，系统自动缓存图片到本地。必须正确设置 `Base Url` 以确保视频能正常显示！**

#### 视频生成模式

在提示词中可以添加 `--mode` 参数来控制视频生成风格：

- `--mode=normal`: 正常模式（默认）
- `--mode=custom`: 自定义模式
- `--mode=extremely-crazy`: 极度疯狂模式
- `--mode=extremely-spicy-or-crazy`: 极度刺激或疯狂模式

#### 视频参数

支持以下可选参数（符合 OpenAI Sora API 格式）：

- `aspect_ratio`: 视频宽高比，如 `"16:9"`, `"2:3"`, `"1:1"` 等
- `duration`: 视频时长（秒），范围 1-60
- `video_length`: 兼容参数，等同于 `duration`（向后兼容）

#### 使用示例

**基础示例：**

```
curl https://你的服务器地址/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "grok-imagine-0.9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "让太阳升起来"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://your-image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

**带模式和参数的示例：**

```
curl https://你的服务器地址/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "grok-imagine-0.9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "让太阳升起来 --mode=extremely-crazy"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://your-image.jpg"
            }
          }
        ]
      }
    ],
    "aspect_ratio": "16:9",
    "duration": 6
  }'
```

**多张图片生成视频示例：**

```
curl https://你的服务器地址/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "grok-imagine-0.9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "将这些图片合成为视频 --mode=normal"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://origin.picgo.net/2025/10/30/image6533bd16d52aff8c.jpg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://origin.picgo.net/2025/10/30/imageb171f51936480634.jpg"
            }
          }
        ]
      }
    ],
    "aspect_ratio": "2:3",
    "duration": 6
  }'
```

### 文本转语音功能（TTS）

- 使用标准的 OpenAI TTS API 格式调用
- 支持将文本转换为语音，返回 WAV 格式音频文件
- **返回方式：非流式，一次性返回完整音频文件**
- **注意：Grok 返回的是 WAV 格式音频，即使请求其他格式也会返回 WAV**

#### 使用示例

**基础示例（curl）：**

```bash
curl https://你的服务器地址/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "tts-1",
    "input": "你好，这是一段测试文本",
    "response_format": "wav"
  }' \
  --output speech.wav
```

**PowerShell 示例：**

```powershell
$body = @{
    model = "tts-1"
    input = "你好，这是一段测试文本"
    response_format = "wav"
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/v1/audio/speech `
    -Method POST `
    -Body $body `
    -ContentType "application/json" `
    -OutFile speech.wav
```

**参数说明：**

- `model` (可选): 模型名称，默认 `"tts-1"`，实际使用 `grok-4.1`
- `input` (必填): 要转换的文本，最大 5000 字符
- `response_format` (可选): 响应格式，默认 `"wav"`（Grok 只返回 WAV 格式）
- `voice` (可选): 语音类型，Grok 可能不支持，保留以兼容 OpenAI 格式
- `speed` (可选): 语速，Grok 可能不支持

### 关于 `x_statsig_id`

- `x_statsig_id` 是 Grok 用于反机器人的 Token，有逆向资料可参考
- **建议新手勿修改配置，保留默认值即可**
- 尝试用 Camoufox 绕过 403 自动获 id，但 grok 现已限制非登陆的`x_statsig_id`，故弃用，采用固定值以兼容所有请求

<br>

## 如何部署

### docker-compose

```yaml
services:
  grok2api:
    image: ghcr.io/chenyme/grok2api:latest
    ports:
      - '8000:8000'
    volumes:
      - grok_data:/app/data
      - ./logs:/app/logs
    environment:
      # =====存储模式: file, mysql 或 redis=====
      - STORAGE_MODE=file
      # =====数据库连接 URL (仅在STORAGE_MODE=mysql或redis时需要)=====
      # - DATABASE_URL=mysql://user:password@host:3306/grok2api

      ## MySQL格式: mysql://user:password@host:port/database
      ## Redis格式: redis://host:port/db 或 redis://user:password@host:port/db (SSL: rediss://)

volumes:
  grok_data:
```

### 环境变量说明

| 环境变量     | 必填 | 说明                                     | 示例                           |
| ------------ | ---- | ---------------------------------------- | ------------------------------ |
| STORAGE_MODE | 否   | 存储模式：file/mysql/redis               | file                           |
| DATABASE_URL | 否   | 数据库连接 URL（MySQL/Redis 模式时必需） | mysql://user:pass@host:3306/db |

**存储模式：**

- `file`: 本地文件存储（默认）
- `mysql`: MySQL 数据库存储，需设置 DATABASE_URL
- `redis`: Redis 缓存存储，需设置 DATABASE_URL

<br>

## 接口说明

> 支持多种 AI 模型接口格式，所有 API 请求均需通过 **Authorization header** 认证

### OpenAI 兼容接口

| 方法 | 端点                   | 描述                        | 是否需要认证 |
| ---- | ---------------------- | --------------------------- | ------------ |
| POST | `/v1/chat/completions` | 创建聊天对话（流式/非流式） | ✅           |
| GET  | `/v1/models`           | 获取全部支持模型            | ✅           |
| POST | `/v1/audio/speech`     | 文本转语音（TTS）           | ✅           |
| GET  | `/images/{img_path}`   | 获取生成图片文件            | ❌           |

### Anthropic 兼容接口

| 方法 | 端点           | 描述                              | 是否需要认证 |
| ---- | -------------- | --------------------------------- | ------------ |
| POST | `/v1/messages` | 创建消息（Anthropic Claude 格式） | ✅           |

> 📖 **使用指南**:
>
> - 快速开始: [ANTHROPIC\_接口说明.md](./ANTHROPIC_接口说明.md)
> - 完整文档: [ANTHROPIC_API.md](./ANTHROPIC_API.md)
> - 测试示例: [ANTHROPIC_DEMO.py](./ANTHROPIC_DEMO.py)

<br>

<details>
<summary>管理与统计接口（展开查看更多）</summary>

| 方法 | 端点                    | 描述              | 认证 |
| ---- | ----------------------- | ----------------- | ---- |
| GET  | /login                  | 管理员登录页面    | ❌   |
| GET  | /manage                 | 管理控制台页面    | ❌   |
| POST | /api/login              | 管理员登录认证    | ❌   |
| POST | /api/logout             | 管理员登出        | ✅   |
| GET  | /api/tokens             | 获取 Token 列表   | ✅   |
| POST | /api/tokens/add         | 批量添加 Token    | ✅   |
| POST | /api/tokens/delete      | 批量删除 Token    | ✅   |
| GET  | /api/settings           | 获取系统配置      | ✅   |
| POST | /api/settings           | 更新系统配置      | ✅   |
| GET  | /api/cache/size         | 获取缓存大小      | ✅   |
| POST | /api/cache/clear        | 清理所有缓存      | ✅   |
| POST | /api/cache/clear/images | 清理图片缓存      | ✅   |
| POST | /api/cache/clear/videos | 清理视频缓存      | ✅   |
| GET  | /api/stats              | 获取统计信息      | ✅   |
| POST | /api/tokens/tags        | 更新 Token 标签   | ✅   |
| POST | /api/tokens/note        | 更新 Token 备注   | ✅   |
| POST | /api/tokens/test        | 测试 Token 可用性 | ✅   |
| GET  | /api/tokens/tags/all    | 获取所有标签列表  | ✅   |
| GET  | /api/storage/mode       | 获取存储模式信息  | ✅   |

</details>

<br>

## 可用模型一览

| 模型名称             | 计次 | 账户类型    | 图像生成/编辑 | 深度思考 | 联网搜索 | 视频生成 |
| -------------------- | ---- | ----------- | ------------- | -------- | -------- | -------- |
| `grok-4.1`           | 1    | Basic/Super | ✅            | ✅       | ✅       | ❌       |
| `grok-4.1-thinking`  | 1    | Basic/Super | ✅            | ✅       | ✅       | ❌       |
| `grok-imagine-0.9`   | -    | Basic/Super | ✅            | ❌       | ❌       | ✅       |
| `grok-4-fast`        | 1    | Basic/Super | ✅            | ✅       | ✅       | ❌       |
| `grok-4-fast-expert` | 4    | Basic/Super | ✅            | ✅       | ✅       | ❌       |
| `grok-4-expert`      | 4    | Basic/Super | ✅            | ✅       | ✅       | ❌       |
| `grok-4-heavy`       | 1    | Super       | ✅            | ✅       | ✅       | ❌       |
| `grok-3-fast`        | 1    | Basic/Super | ✅            | ❌       | ✅       | ❌       |

<br>

## 配置参数说明

> 服务启动后，登录 `/login` 管理后台进行参数配置

| 参数名                        | 作用域 | 必填 | 说明                                | 默认值                                                                                             |
| ----------------------------- | ------ | ---- | ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| admin_username                | global | 否   | 管理后台登录用户名                  | "admin"                                                                                            |
| admin_password                | global | 否   | 管理后台登录密码                    | "admin"                                                                                            |
| log_level                     | global | 否   | 日志级别：DEBUG/INFO/...            | "INFO"                                                                                             |
| image_mode                    | global | 否   | 图片返回模式：url/base64            | "url"                                                                                              |
| image_cache_max_size_mb       | global | 否   | 图片缓存最大容量(MB)                | 512                                                                                                |
| video_cache_max_size_mb       | global | 否   | 视频缓存最大容量(MB)                | 1024                                                                                               |
| base_url                      | global | 否   | 服务基础 URL/图片访问基准           | ""                                                                                                 |
| api_key                       | grok   | 否   | API 密钥（可选加强安全）            | ""                                                                                                 |
| proxy_url                     | grok   | 否   | HTTP 代理服务器地址                 | ""                                                                                                 |
| stream_chunk_timeout          | grok   | 否   | 流式分块超时时间(秒)                | 120                                                                                                |
| stream_first_response_timeout | grok   | 否   | 流式首次响应超时时间(秒)            | 30                                                                                                 |
| stream_total_timeout          | grok   | 否   | 流式总超时时间(秒)                  | 600                                                                                                |
| cf_clearance                  | grok   | 否   | Cloudflare 安全令牌                 | ""                                                                                                 |
| x_statsig_id                  | grok   | 是   | 反机器人唯一标识符                  | "ZTpUeXBlRXJyb3I6IENhbm5vdCByZWFkIHByb3BlcnRpZXMgb2YgdW5kZWZpbmVkIChyZWFkaW5nICdjaGlsZE5vZGVzJyk=" |
| filtered_tags                 | grok   | 否   | 过滤响应标签（逗号分隔）            | "xaiartifact,xai:tool_usage_card,grok:render"                                                      |
| show_thinking                 | grok   | 否   | 显示思考过程 true(显示)/false(隐藏) | true                                                                                               |
| temporary                     | grok   | 否   | 会话模式 true(临时)/false           | true                                                                                               |

<br>

## ⚠️ 注意事项

本项目仅供学习与研究，请遵守相关使用条款！

<br>

> 本项目基于以下项目学习重构，特别感谢：[LINUX DO](https://linux.do)、[VeroFess/grok2api](https://github.com/VeroFess/grok2api)、[xLmiler/grok2api_python](https://github.com/xLmiler/grok2api_python)
