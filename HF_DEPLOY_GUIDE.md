# Hugging Face Space 部署指南

本文档介绍如何将 Grok2API 部署到 Hugging Face Space。

## 快速部署

### 方法一：直接复制文件（推荐）

1. **创建新的 HF Space**
   - 访问 https://huggingface.co/new-space
   - 输入 Space 名称，如 `grok2api`
   - 选择 **Docker** 作为 SDK
   - 选择可见性（建议 Private）
   - 点击 Create Space

2. **克隆 Space 仓库**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/grok2api
   cd grok2api
   ```

3. **复制项目文件**
   ```bash
   # 复制核心文件
   cp -r /path/to/grok2api/app ./
   cp /path/to/grok2api/main.py ./
   cp /path/to/grok2api/requirements.txt ./
   
   # 复制 HF 专用文件并重命名
   cp /path/to/grok2api/Dockerfile.hf ./Dockerfile
   cp /path/to/grok2api/docker-entrypoint-hf.sh ./docker-entrypoint-hf.sh
   cp /path/to/grok2api/HF_SPACE_README.md ./README.md
   ```

4. **推送到 HF**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

### 方法二：使用 GitHub 同步

1. 将项目推送到 GitHub（私有仓库）
2. 在 HF Space 设置中连接 GitHub 仓库
3. 确保仓库根目录有正确的 `Dockerfile` 和 `README.md`

## 配置环境变量

部署后，在 Space Settings -> Repository secrets 中添加：

```
ADMIN_USERNAME=your_admin_name
ADMIN_PASSWORD=your_secure_password
API_KEY=your_api_key
BASE_URL=https://your-username-grok2api.hf.space
```

### 可选：使用外部数据库

```
STORAGE_MODE=mysql
DATABASE_URL=mysql://user:password@host:3306/grok2api
```

或 Redis：

```
STORAGE_MODE=redis
DATABASE_URL=redis://host:6379/0
```

## 文件清单

部署到 HF Space 需要以下文件：

```
your-space/
├── README.md              # 从 HF_SPACE_README.md 复制
├── Dockerfile             # 从 Dockerfile.hf 复制
├── docker-entrypoint-hf.sh
├── requirements.txt
├── main.py
└── app/
    ├── api/
    ├── core/
    ├── models/
    ├── services/
    └── template/
```

## 验证部署

1. 等待 Space 构建完成（可能需要几分钟）
2. 访问 `https://your-username-grok2api.hf.space/health` 检查健康状态
3. 访问 `https://your-username-grok2api.hf.space/login` 登录管理后台

## 常见问题

### Q: 构建失败，提示 curl_cffi 安装错误

确保使用 `Dockerfile.hf` 而不是原版 `Dockerfile`，HF 版本包含了必要的系统依赖。

### Q: 重启后数据丢失

这是免费版 Space 的限制。解决方案：
- 使用外部 MySQL/Redis 存储
- 升级到付费 Space（有持久化存储）

### Q: API 访问返回 401

检查是否设置了 `API_KEY` 环境变量，以及请求头中是否包含正确的 `Authorization: Bearer YOUR_API_KEY`。

### Q: 图片/视频无法显示

确保 `BASE_URL` 设置正确，应为完整的 Space URL：
```
https://your-username-grok2api.hf.space
```

## 资源限制

| 类型 | 免费版 | 付费版 |
|------|--------|--------|
| CPU | 2 vCPU | 更多 |
| 内存 | 16 GB | 更多 |
| 存储 | 临时 | 持久化 |
| 休眠 | 48h 无活动后休眠 | 可配置 |

## 安全建议

1. 将 Space 设为 Private
2. 使用强密码
3. 设置 API_KEY 限制访问
4. 定期更换 Token
