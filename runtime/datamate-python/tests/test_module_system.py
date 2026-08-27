import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import select

from app.db.datascope import DataScopeHandle
from app.db.models.dataset_management import Dataset
from app.db.models.models import Models
from app.db.session import _apply_data_scope
from app.module.system.service.common_service import get_model_by_id


def _run(coro):
    return asyncio.run(coro)


def test_get_model_by_id_returns_model_when_found() -> None:
    db = MagicMock()
    model = SimpleNamespace(id="m1")
    result = MagicMock()
    result.scalar_one_or_none.return_value = model
    db.execute = AsyncMock(return_value=result)

    fetched = _run(get_model_by_id(db, "m1"))

    assert fetched is model


def test_get_model_by_id_returns_none_when_missing() -> None:
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    fetched = _run(get_model_by_id(db, "missing"))

    assert fetched is None


def test_get_model_by_id_invokes_db_execute_once() -> None:
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    _run(get_model_by_id(db, "m2"))

    db.execute.assert_called_once()


def test_get_model_by_id_passes_query_object_to_execute() -> None:
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)

    _run(get_model_by_id(db, "model-xyz"))

    args, _ = db.execute.call_args
    assert len(args) == 1
    assert args[0] is not None


def test_get_model_by_id_returns_exact_scalar_object() -> None:
    db = MagicMock()
    model_obj = SimpleNamespace(id="m100", endpoint="x")
    result = MagicMock()
    result.scalar_one_or_none.return_value = model_obj
    db.execute = AsyncMock(return_value=result)

    fetched = _run(get_model_by_id(db, "m100"))
    assert fetched is model_obj


def _compile_scoped_query(model, user: str) -> str:
    DataScopeHandle.set_user_info(user)
    try:
        state = SimpleNamespace(is_select=True, statement=select(model))
        _apply_data_scope(state)
        return str(state.statement.compile(compile_kwargs={"literal_binds": True}))
    finally:
        DataScopeHandle.remove_user_info()


def test_regular_user_model_query_keeps_creator_scope() -> None:
    sql = _compile_scoped_query(Models, "alice")

    assert "t_models.created_by IN ('alice', 'system')" in sql


def test_admin_model_query_bypasses_creator_scope() -> None:
    sql = _compile_scoped_query(Models, "admin")

    assert "t_models.created_by IN" not in sql


def test_similar_username_does_not_gain_admin_model_access() -> None:
    sql = _compile_scoped_query(Models, "Admin")

    assert "t_models.created_by IN ('Admin', 'system')" in sql


def test_system_user_does_not_gain_admin_model_access() -> None:
    sql = _compile_scoped_query(Models, "system")

    assert "t_models.created_by IN ('system', 'system')" in sql


def test_admin_query_keeps_scope_for_unmarked_models() -> None:
    sql = _compile_scoped_query(Dataset, "admin")

    assert "t_dm_datasets.created_by IN ('admin', 'system')" in sql
