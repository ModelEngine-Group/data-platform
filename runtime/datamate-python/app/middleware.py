"""
Application middleware.

Note: Exception handling has been moved to app.core.exception.ExceptionHandlingMiddleware
This file now only contains the UserContextMiddleware.
"""
import json
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED

from app.core.config import settings
from app.core.logging import get_logger
from app.db.datascope import DataScopeHandle

logger = get_logger(__name__)


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    读取 `User` 请求头并设置 DataScopeHandle 的 FastAPI 中间件。
    如果 `jwt_enable` 为 True，缺少请求头将返回 401。
    """

    def __init__(self, app):
        super().__init__(app)
        self.jwt_enable = settings.datamate_jwt_enable

    async def dispatch(self, request: Request, call_next):
        user: Optional[str] = request.headers.get("User")
        logger.info(f"start filter, current user: {user}, need filter: {self.jwt_enable}")
        if self.jwt_enable and (user is None or user.strip() == ""):
            payload = {"code": "common.401", "message": "unauthorized", "data": None}
            return Response(
                content=json.dumps(payload),
                status_code=HTTP_401_UNAUTHORIZED,
                media_type="application/json"
            )

        DataScopeHandle.set_user_info(user)
        try:
            response = await call_next(request)
            return response
        finally:
            DataScopeHandle.remove_user_info()


# Re-export ExceptionHandlingMiddleware for backward compatibility

__all__ = ['UserContextMiddleware']
