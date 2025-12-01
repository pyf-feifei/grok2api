"""异常处理器 - OpenAI兼容的错误响应"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# HTTP错误映射
HTTP_ERROR_MAP = {
    400: ("invalid_request_error", "请求格式错误或缺少必填参数"),
    401: ("invalid_request_error", "令牌认证失败"),
    403: ("permission_error", "没有权限访问此资源"),
    404: ("invalid_request_error", "请求的资源不存在"),
    429: ("rate_limit_error", "请求频率超出限制，请稍后再试"),
    500: ("api_error", "内部服务器错误"),
    503: ("api_error", "服务暂时不可用"),
}

# Grok错误码映射
GROK_STATUS_MAP = {
    "NO_AUTH_TOKEN": status.HTTP_401_UNAUTHORIZED,
    "INVALID_TOKEN": status.HTTP_401_UNAUTHORIZED,
    "HTTP_ERROR": status.HTTP_502_BAD_GATEWAY,
    "NETWORK_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    "JSON_ERROR": status.HTTP_502_BAD_GATEWAY,
    "API_ERROR": status.HTTP_502_BAD_GATEWAY,
    "STREAM_ERROR": status.HTTP_502_BAD_GATEWAY,
    "NO_RESPONSE": status.HTTP_502_BAD_GATEWAY,
    "TOKEN_SAVE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "NO_AVAILABLE_TOKEN": status.HTTP_503_SERVICE_UNAVAILABLE,
}

GROK_TYPE_MAP = {
    "NO_AUTH_TOKEN": "authentication_error",
    "INVALID_TOKEN": "authentication_error",
    "HTTP_ERROR": "api_error",
    "NETWORK_ERROR": "api_error",
    "JSON_ERROR": "api_error",
    "API_ERROR": "api_error",
    "STREAM_ERROR": "api_error",
    "NO_RESPONSE": "api_error",
    "TOKEN_SAVE_ERROR": "api_error",
    "NO_AVAILABLE_TOKEN": "api_error",
}


class GrokApiException(Exception):
    """Grok API业务异常"""

    def __init__(self, message: str, error_code: str = None, details: dict = None, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.context = context or {}
        super().__init__(self.message)


def build_error_response(message: str, error_type: str, code: str = None, param: str = None) -> dict:
    """构建OpenAI兼容的错误响应"""
    error = {"message": message, "type": error_type}
    
    if code:
        error["code"] = code
    if param:
        error["param"] = param

    return {"error": error}


def build_anthropic_error_response(error_type: str, message: str) -> dict:
    """构建 Anthropic 兼容的错误响应"""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """处理HTTP异常"""
    # 如果 detail 是字典，检查具体格式
    if isinstance(exc.detail, dict):
        # 检查是否是 Anthropic 格式的错误（来自 anthropic.py）
        if "type" in exc.detail and exc.detail.get("type") == "error":
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail
            )
        
        # 检查是否是认证错误（来自 auth.py）- error 是字典
        if "error" in exc.detail and isinstance(exc.detail["error"], dict):
            error_info = exc.detail["error"]
            error_type = error_info.get("type", "api_error")
            message = error_info.get("message", "未知错误")
            # 返回 Anthropic 兼容格式
            return JSONResponse(
                status_code=exc.status_code,
                content=build_anthropic_error_response(error_type, message)
            )
        
        # 其他字典格式（如 admin 接口）- error 是字符串
        if "error" in exc.detail and isinstance(exc.detail["error"], str):
            message = exc.detail["error"]
            error_type, _ = HTTP_ERROR_MAP.get(exc.status_code, ("api_error", message))
            return JSONResponse(
                status_code=exc.status_code,
                content=build_error_response(message, error_type, exc.detail.get("code"))
            )
    
    # 默认处理
    error_type, default_msg = HTTP_ERROR_MAP.get(exc.status_code, ("api_error", str(exc.detail)))
    message = str(exc.detail) if exc.detail else default_msg

    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_response(message, error_type)
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """处理验证错误"""
    errors = exc.errors()
    param = errors[0]["loc"][-1] if errors and errors[0].get("loc") else None
    message = errors[0]["msg"] if errors and errors[0].get("msg") else "请求参数错误"

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=build_error_response(message, "invalid_request_error", param=param)
    )


async def grok_api_exception_handler(_: Request, exc: GrokApiException) -> JSONResponse:
    """处理Grok API异常"""
    http_status = GROK_STATUS_MAP.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    error_type = GROK_TYPE_MAP.get(exc.error_code, "api_error")

    return JSONResponse(
        status_code=http_status,
        content=build_error_response(exc.message, error_type, exc.error_code)
    )


async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """处理未捕获异常"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=build_error_response("服务器遇到意外错误，请重试", "api_error")
    )


def register_exception_handlers(app) -> None:
    """注册异常处理器"""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(GrokApiException, grok_api_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)