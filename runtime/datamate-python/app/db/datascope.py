from contextvars import ContextVar
from typing import List, Optional

_current_user: ContextVar[Optional[str]] = ContextVar("_current_user", default=None)
SYSTEM_USER = "system"
ADMIN_USER = "admin"


class DataScopeHandle:
    """
    Hold current user info in a ContextVar and provide helpers for SQLAlchemy filters.
    """

    @staticmethod
    def set_user_info(user: Optional[str]) -> None:
        if user is None or user == "":
            # set explicit None
            _current_user.set(None)
        else:
            _current_user.set(user)

    @staticmethod
    def remove_user_info() -> None:
        _current_user.set(None)

    @staticmethod
    def get_user_info() -> Optional[str]:
        return _current_user.get()

    @staticmethod
    def is_admin() -> bool:
        return DataScopeHandle.get_user_info() == ADMIN_USER

    @staticmethod
    def allowed_users() -> List[str]:
        """
        Return list of allowed creators: current user + system.
        """
        user = DataScopeHandle.get_user_info()
        if not user:
            return []
        return [user, SYSTEM_USER]
