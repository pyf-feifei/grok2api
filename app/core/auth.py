"""认证模块 - API令牌验证"""

from typing import Optional
from fastapi import Depends, HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import setting
from app.core.logger import logger


# Bearer安全方案
security = HTTPBearer(auto_error=False)


def _build_error(message: str, code: str = "invalid_token") -> dict:
    """构建认证错误"""
    return {
        "error": {
            "message": message,
            "type": "authentication_error",
            "code": code
        }
    }


class AuthManager:
    """认证管理器 - 验证API令牌（支持 Bearer 和 x-api-key）"""

    @staticmethod
    def verify(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            security),
        x_api_key: Optional[str] = Header(None, alias="x-api-key")
    ) -> Optional[str]:
        """验证令牌 - 支持 Authorization: Bearer 和 x-api-key header"""
        api_key = setting.grok_config.get("api_key")

        # 详细日志 - 记录请求头
        auth_header = request.headers.get("authorization", "")
        x_api_key_header = request.headers.get("x-api-key", "")
        logger.debug(f"[Auth] 请求路径: {request.url.path}")
        logger.debug(f"[Auth] Authorization header: {auth_header[:50]}..." if len(
            auth_header) > 50 else f"[Auth] Authorization header: {auth_header or '(空)'}")
        logger.debug(f"[Auth] x-api-key header: {x_api_key_header[:50]}..." if len(
            x_api_key_header) > 50 else f"[Auth] x-api-key header: {x_api_key_header or '(空)'}")
        logger.debug(f"[Auth] 服务器配置的 api_key: {api_key[:20]}..." if api_key and len(
            api_key) > 20 else f"[Auth] 服务器配置的 api_key: {api_key or '(未配置)'}")

        # 未设置时跳过
        if not api_key:
            logger.debug("[Auth] 未设置API_KEY，跳过验证")
            # 返回找到的任意token
            if credentials:
                return credentials.credentials
            if x_api_key:
                return x_api_key
            return None

        # 获取token（优先使用 Bearer，其次 x-api-key）
        token = None
        if credentials:
            token = credentials.credentials
            logger.debug(f"[Auth] 使用 Bearer token")
        elif x_api_key:
            token = x_api_key
            logger.debug(f"[Auth] 使用 x-api-key")

        # 检查令牌
        if not token:
            logger.warning(f"[Auth] 缺少认证令牌 - 请求路径: {request.url.path}")
            raise HTTPException(
                status_code=401,
                detail=_build_error("缺少认证令牌", "missing_token")
            )

        # 验证令牌
        if token != api_key:
            logger.warning(
                f"[Auth] 令牌无效 - 收到: {token[:20]}..., 期望: {api_key[:20]}...")
            raise HTTPException(
                status_code=401,
                detail=_build_error(f"令牌无效", "invalid_token")
            )

        logger.debug("[Auth] 令牌认证成功")
        return token


# 全局实例
auth_manager = AuthManager()
