import math
import uuid
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.module.shared.llm import LLMFactory
from app.core.logging import get_logger
from app.db.datascope import DataScopeHandle, SYSTEM_USER
from app.db.models.models import Models
from app.db.session import get_db
from app.module.shared.schema import PaginatedData
from app.module.system.schema.models import (
    CreateModelRequest,
    QueryModelRequest,
    ModelsResponse,
    ProviderItem,
)

logger = get_logger(__name__)

# 固定厂商列表
PROVIDERS = [
    ProviderItem(provider="ModelEngine", baseUrl="http://nginxservice/open/router/v1"),
    ProviderItem(provider="Ollama", baseUrl="http://localhost:11434"),
    ProviderItem(provider="OpenAI", baseUrl="https://api.openai.com/v1"),
    ProviderItem(provider="DeepSeek", baseUrl="https://api.deepseek.com/v1"),
    ProviderItem(provider="火山方舟", baseUrl="https://ark.cn-beijing.volces.com/api/v3"),
    ProviderItem(provider="阿里云百炼", baseUrl="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ProviderItem(provider="硅基流动", baseUrl="https://api.siliconflow.cn/v1"),
    ProviderItem(provider="智谱AI", baseUrl="https://open.bigmodel.cn/api/paas/v4"),
    ProviderItem(provider="自定义模型", baseUrl=""),
]


def _created_by_match(user: str):
    return (
        (Models.created_by == user)
        | ((Models.created_by.is_(None)) & (user == SYSTEM_USER))
    )


def _orm_to_response(row: Models) -> ModelsResponse:
    return ModelsResponse(
        id=row.id,
        modelName=row.model_name,
        provider=row.provider,
        baseUrl=row.base_url,
        apiKey=row.api_key or "",
        type=row.type or "CHAT",
        isEnabled=bool(row.is_enabled) if row.is_enabled is not None else True,
        isDefault=bool(row.is_default) if row.is_default is not None else False,
        createdAt=row.created_at,
        updatedAt=row.updated_at,
        createdBy=row.created_by,
        updatedBy=row.updated_by,
    )


class ModelsService:
    """模型配置服务：db 通过 FastAPI Depends(get_db) 注入，不在路由中传递。"""

    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def get_providers(self) -> list[ProviderItem]:
        """返回固定厂商列表，与 Java getProviders() 一致。"""
        return list(PROVIDERS)

    async def get_models(self, q: QueryModelRequest) -> PaginatedData[ModelsResponse]:
        """分页查询，支持 provider/type/isEnabled/isDefault；排除已删除；page 从 0 开始。"""
        query = select(Models).where(
            (Models.is_deleted == False) | (Models.is_deleted.is_(None))
        )
        if q.provider is not None and q.provider != "":
            query = query.where(Models.provider == q.provider)
        if q.type is not None:
            query = query.where(Models.type == q.type.value)
        if q.isEnabled is not None:
            query = query.where(Models.is_enabled == q.isEnabled)
        if q.isDefault is not None:
            query = query.where(Models.is_default == q.isDefault)

        total = (await self.db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
        size = max(1, min(500, q.size))
        offset = max(0, q.page) * size
        rows = (
            await self.db.execute(
                query.order_by(Models.created_at.desc()).offset(offset).limit(size)
            )
        ).scalars().all()
        total_pages = math.ceil(total / size) if total else 0
        current = max(0, q.page) + 1
        return PaginatedData(
            page=current,
            size=size,
            total_elements=total,
            total_pages=total_pages,
            content=[_orm_to_response(r) for r in rows],
        )

    async def get_model_detail(self, model_id: str) -> ModelsResponse:
        """获取模型详情，已删除或不存在则 404。"""
        query = select(Models).where(Models.id == model_id).where(
            (Models.is_deleted == False) | (Models.is_deleted.is_(None))
        )
        r = (await self.db.execute(query)).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="模型配置不存在")
        return _orm_to_response(r)

    async def create_model(self, req: CreateModelRequest) -> ModelsResponse:
        """创建模型：健康检查后 saveAndSetDefault；isEnabled 恒为 True。
        若 uk_model_provider 已存在且 is_deleted=True，则恢复并更新相关字段。"""
        try:
            LLMFactory.check_health(req.modelName, req.baseUrl, req.apiKey, req.type.value)
        except Exception as e:
            logger.error("Model health check failed: model=%s type=%s err=%s", req.modelName, req.type, e)
            raise HTTPException(status_code=400, detail="模型健康检查失败") from e

        # 检查 uk_model_provider (model_name, provider, created_by) 是否存在且已删除
        effective_user = (DataScopeHandle.get_user_info() or "").strip() or SYSTEM_USER
        created_by_match = _created_by_match(effective_user)
        deleted_row = (
            await self.db.execute(
                select(Models).where(
                    Models.model_name == req.modelName,
                    Models.provider == req.provider,
                    created_by_match,
                )
            )
        ).scalar_one_or_none()

        if deleted_row and deleted_row.is_deleted:
            # 恢复并更新
            entity = deleted_row
            entity.base_url = req.baseUrl
            entity.api_key = req.apiKey or ""
            entity.type = req.type.value
            entity.is_enabled = True
            entity.is_deleted = False
            entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            want_default = req.isDefault if req.isDefault is not None else False
            if want_default:
                await self.db.execute(
                    update(Models)
                    .where(
                        Models.type == req.type.value,
                        Models.is_default == True,
                        created_by_match,
                    )
                    .values(is_default=False)
                )
            entity.is_default = want_default

            self.db.add(entity)
            await self.db.commit()
            await self.db.refresh(entity)
            return _orm_to_response(entity)

        # 正常创建
        existing = (
            await self.db.execute(
                select(Models).where(
                    (Models.is_deleted == False) | (Models.is_deleted.is_(None)),
                    Models.type == req.type.value,
                    Models.is_default == True,
                    created_by_match,
                )
            )
        ).scalar_one_or_none()

        is_default: bool
        if existing is None:
            is_default = True
        else:
            await self.db.execute(
                update(Models)
                .where(
                    Models.type == req.type.value,
                    Models.is_default == True,
                    created_by_match,
                )
                .values(is_default=False)
            )
            is_default = req.isDefault if req.isDefault is not None else False

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        entity = Models(
            id=str(uuid.uuid4()),
            model_name=req.modelName,
            provider=req.provider,
            base_url=req.baseUrl,
            api_key=req.apiKey or "",
            type=req.type.value,
            is_enabled=True,
            is_default=is_default,
            is_deleted=False,
            created_at=now,
            updated_at=now,
        )
        self.db.add(entity)
        await self.db.commit()
        await self.db.refresh(entity)
        return _orm_to_response(entity)

    async def update_model(self, model_id: str, req: CreateModelRequest) -> ModelsResponse:
        """更新模型：存在性校验、健康检查后 updateAndSetDefault；isEnabled 恒为 True。"""
        res = await self.db.execute(
            select(Models).where(Models.id == model_id).where(
                (Models.is_deleted == False) | (Models.is_deleted.is_(None))
            )
        )
        entity = res.scalar_one_or_none()
        if not entity:
            raise HTTPException(status_code=404, detail="模型配置不存在")

        try:
            LLMFactory.check_health(req.modelName, req.baseUrl, req.apiKey, req.type.value)
        except Exception as e:
            logger.error("Model health check failed: model=%s type=%s err=%s", req.modelName, req.type, e)
            raise HTTPException(status_code=400, detail="模型健康检查失败") from e

        entity.model_name = req.modelName
        entity.provider = req.provider
        entity.base_url = req.baseUrl
        entity.api_key = req.apiKey or ""
        entity.type = req.type.value
        entity.is_enabled = True

        want_default = req.isDefault if req.isDefault is not None else False
        if (entity.is_default is not True) and want_default:
            owner = (entity.created_by or "").strip() or SYSTEM_USER
            await self.db.execute(
                update(Models)
                .where(
                    Models.type == req.type.value,
                    Models.is_default == True,
                    _created_by_match(owner),
                )
                .values(is_default=False)
            )
        entity.is_default = want_default
        entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        await self.db.commit()
        await self.db.refresh(entity)
        return _orm_to_response(entity)

    async def delete_model(self, model_id: str) -> None:
        """软删除模型配置。"""
        entity = (
            await self.db.execute(
                select(Models).where(Models.id == model_id).where(
                    (Models.is_deleted == False) | (Models.is_deleted.is_(None))
                )
            )
        ).scalar_one_or_none()
        if not entity:
            raise HTTPException(status_code=404, detail="模型配置不存在")
        entity.is_deleted = True
        entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(entity)
