from typing import Optional
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_, select

from app.db.session import get_db
from app.db.models import LabelingProject
from app.module.shared.schema import StandardResponse, PaginatedData
from app.module.dataset import DatasetManagementService
from app.core.logging import get_logger
from app.core.config import settings

from ..client import LabelStudioClient
from ..service.mapping import DatasetMappingService
from ..service.sync import SyncService
from ..service.ls_annotation_sync import LSAnnotationSyncService
from ..service.template import AnnotationTemplateService
from ..schema import (
    DatasetMappingCreateRequest,
    DatasetMappingCreateResponse,
    DeleteDatasetResponse,
    DatasetMappingResponse,
)
from ..schema.auto import ImportFromLabelStudioRequest, UpdateAutoAnnotationTaskFilesRequest

router = APIRouter(
    prefix="/project",
    tags=["annotation/project"]
)
logger = get_logger(__name__)

@router.get("/{mapping_id}/login")
async def login_label_studio_project(
    mapping_id: str = Path(..., description="项目映射ID"),
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """登录 Label Studio 项目 — 需要认证且校验项目归属"""
    # 检查用户认证（仅在 JWT 启用时强制要求）
    user = request.headers.get("User") if request else None
    if settings.datamate_jwt_enable and (not user or not user.strip()):
        logger.warning(f"Unauthorized access attempt to annotation login: mapping_id={mapping_id}")
        raise HTTPException(status_code=401, detail="Authentication required")

    # 兼容两类调用方：
    # - DataMate 映射 UUID（当前接口约定）
    # - Label Studio 项目 ID（旧版前端仍可能传入）
    # 两者最终都需要定位到同一条 LabelingProject 映射记录。
    stmt = select(LabelingProject).where(
        or_(
            LabelingProject.id == mapping_id,
            LabelingProject.labeling_project_id == mapping_id,
        ),
        LabelingProject.deleted_at.is_(None)
    )
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        logger.warning(f"Project not found for login: mapping_id={mapping_id}, user={user or 'anonymous'}")
        raise HTTPException(status_code=404, detail="Project not found")

    logger.info(f"User '{user or 'anonymous'}' logging into Label Studio project: mapping_id={mapping_id}, "
                f"project_name={project.name}")

    try:
        ls_client = LabelStudioClient(base_url=settings.label_studio_base_url,
                                      token=settings.label_studio_user_token)
        target_response = await ls_client.login_label_studio()
        if target_response is None:
            raise HTTPException(status_code=502, detail="Label Studio login failed")

        headers = dict(target_response.headers)
        set_cookies = target_response.headers.get_list("set-cookie")

        # 删除合并的 Set-Cookie
        if "set-cookie" in headers:
            del headers["set-cookie"]

        # 创建新响应，添加多个 Set-Cookie
        response = Response(
            content=target_response.content,
            status_code=target_response.status_code,
            headers=headers
        )

        # 分别添加每个 Set-Cookie
        for cookie in set_cookies:
            response.headers.append("set-cookie", cookie)

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error while logining in LabelStudio: {e}", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("", response_model=StandardResponse[DatasetMappingCreateResponse])
async def create_mapping(
    request: DatasetMappingCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    创建数据集映射

    根据指定的DM程序中的数据集，创建Label Studio中的数据集，
    在数据库中记录这一关联关系，返回Label Studio数据集的ID

    注意：一个数据集可以创建多个标注项目

    支持通过 template_id 指定标注模板，如果提供了模板ID，则使用模板的配置
    """
    try:
        dm_client = DatasetManagementService(db)
        ls_client = LabelStudioClient(
            base_url=settings.label_studio_base_url,
            token=settings.label_studio_user_token,
        )
        mapping_service = DatasetMappingService(db)
        sync_service = SyncService(dm_client, ls_client, mapping_service)
        template_service = AnnotationTemplateService()

        logger.info(
            "Create dataset mapping request: dataset_id=%s, file_ids=%s",
            request.dataset_id,
            request.file_ids,
        )

        # 从DM服务获取数据集信息
        dataset_info = await dm_client.get_dataset(request.dataset_id)
        if not dataset_info:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset not found in DM service: {request.dataset_id}",
            )

        project_name = (
            request.name
            or dataset_info.name
            or "A new project from DataMate"
        )

        project_description = (
            request.description
            or dataset_info.description
            or f"Imported from DM dataset {dataset_info.name} ({dataset_info.id})"
        )

        # 如果提供了模板ID，获取模板配置
        label_config = None
        if request.template_id:
            logger.info("Using template: %s", request.template_id)
            template = await template_service.get_template(db, request.template_id)
            if not template:
                raise HTTPException(
                    status_code=404,
                    detail=f"Template not found: {request.template_id}",
                )
            label_config = template.label_config
            logger.debug(
                "Template label config loaded for template: %s",
                template.name,
            )

        # 在Label Studio中创建项目
        project_data = await ls_client.create_project(
            title=project_name,
            description=project_description,
            label_config=label_config,
        )
        if not project_data:
            raise HTTPException(
                status_code=500,
                detail="Fail to create Label Studio project.",
            )

        project_id = project_data["id"]

        # 配置主数据集的本地存储：dataset/<id>
        local_storage_path = (
            f"{settings.label_studio_local_document_root}/{request.dataset_id}"
        )
        storage_result = await ls_client.create_local_storage(
            project_id=project_id,
            path=local_storage_path,
            title="Dataset_BLOB",
            use_blob_urls=True,
            description=f"Local storage for dataset {dataset_info.name}",
        )

        if not storage_result:
            # 本地存储配置失败，记录警告但不中断流程
            logger.warning(
                "Failed to configure local storage for project %s",
                project_id,
            )
        else:
            logger.info(
                "Local storage configured for project %s: %s",
                project_id,
                local_storage_path,
            )

        labeling_project = LabelingProject(
            id=str(uuid.uuid4()),
            dataset_id=request.dataset_id,
            labeling_project_id=str(project_id),
            name=project_name,
            template_id=request.template_id,
        )

        # 创建映射关系，包含项目名称（先持久化映射以获得 mapping.id）
        mapping = await mapping_service.create_mapping(labeling_project)

        # 如果未指定 file_ids，保持原有行为：按映射数据集完整同步
        if not request.file_ids:
            await sync_service.sync_dataset_files(mapping.id, 100)
        else:
            # 仿照自动标注逻辑：根据 file_ids 反查所属数据集，可跨多个数据集，
            # 最终把这些文件同步到同一个 Label Studio 项目中。
            try:
                from typing import Set as _Set, Dict as _Dict
                from app.db.models.dataset_management import DatasetFiles

                file_ids = request.file_ids

                stmt = (
                    select(DatasetFiles.dataset_id, DatasetFiles.id)
                    .where(DatasetFiles.id.in_(file_ids))
                )
                result = await db.execute(stmt)
                rows = result.fetchall()

                grouped: _Dict[str, _Set[str]] = {}
                resolved_ids: _Set[str] = set()

                for ds_id, fid in rows:
                    if not ds_id or not fid:
                        continue
                    fid_str = str(fid)
                    grouped.setdefault(str(ds_id), set()).add(fid_str)
                    resolved_ids.add(fid_str)

                # 未能解析到数据集的文件，全部归入主数据集，避免丢失
                unresolved_ids = {str(fid) for fid in file_ids} - resolved_ids
                if unresolved_ids:
                    logger.warning(
                        "Some file_ids could not be resolved to dataset_id when syncing manual project files: %s",
                        ",".join(sorted(unresolved_ids)),
                    )
                    grouped.setdefault(str(request.dataset_id), set()).update(
                        unresolved_ids
                    )

                # 为所有涉及到的额外数据集提前配置本地存储
                try:
                    for extra_ds_id in grouped.keys():
                        # 主数据集已在上方配置过
                        if str(extra_ds_id) == str(request.dataset_id):
                            continue

                        extra_local_storage_path = (
                            f"{settings.label_studio_local_document_root}/{extra_ds_id}"
                        )
                        extra_storage_result = await ls_client.create_local_storage(
                            project_id=project_id,
                            path=extra_local_storage_path,
                            title=f"Dataset_BLOB_{extra_ds_id}",
                            use_blob_urls=True,
                            description=(
                                f"Local storage for dataset {extra_ds_id} "
                                "(multi-dataset manual project)"
                            ),
                        )
                        if not extra_storage_result:
                            logger.warning(
                                "Failed to configure extra local storage for project %s (dataset %s)",
                                project_id,
                                extra_ds_id,
                            )
                        else:
                            logger.info(
                                "Extra local storage configured for project %s: %s",
                                project_id,
                                extra_local_storage_path,
                            )
                except Exception as e:  # pragma: no cover - 容错
                    logger.warning(
                        "Error while configuring extra local storage for project %s: %s",
                        project_id,
                        e,
                    )

                if not grouped:
                    # 极端情况：完全无法解析，退回到仅按主数据集＋给定 file_ids 同步
                    await sync_service.sync_files(
                        mapping,
                        100,
                        allowed_file_ids={str(fid) for fid in file_ids},
                        delete_orphans=False,
                    )
                else:
                    # 对每个涉及到的数据集，使用 override_dataset_id 将其文件同步到同一个项目
                    for ds_id, ds_file_ids in grouped.items():
                        await sync_service.sync_files(
                            mapping,
                            100,
                            allowed_file_ids=ds_file_ids,
                            override_dataset_id=ds_id,
                            delete_orphans=False,
                        )
            except Exception as e:  # pragma: no cover - 映射创建成功但首次文件同步失败
                # 同步失败不影响项目和映射本身的创建，前端可通过“同步”按钮重试
                logger.warning(
                    "Failed to sync dataset files for manual LS project %s with file_ids filter: %s",
                    project_id,
                    e,
                )

        response_data = DatasetMappingCreateResponse(
            id=mapping.id,
            labeling_project_id=str(mapping.labeling_project_id),
            labeling_project_name=mapping.name or project_name,
        )

        return StandardResponse(
            code="0",
            message="success",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - 兜底错误
        logger.error(f"Error while creating dataset mapping: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{mapping_id}/files", response_model=StandardResponse[list[dict]])
async def get_manual_mapping_files(
    mapping_id: str = Path(..., description="映射ID (mapping UUID)"),
    db: AsyncSession = Depends(get_db),
):
    """查询手动标注映射当前在 Label Studio 中已存在的 DM 文件列表。

    该接口主要用于前端“编辑任务数据集”弹窗的初始选中状态：
    - 通过 Label Studio 任务反查当前已有关联的 DM 文件ID；
    - 再到 DM 数据库中查询这些文件的基础信息与所属数据集。
    """

    from sqlalchemy import select as _select  # 本地导入避免循环依赖
    from app.db.models.dataset_management import DatasetFiles, Dataset

    mapping_service = DatasetMappingService(db)
    mapping = await mapping_service.get_mapping_by_uuid(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    dm_client = DatasetManagementService(db)
    ls_client = LabelStudioClient(
        base_url=settings.label_studio_base_url,
        token=settings.label_studio_user_token,
    )
    sync_service = SyncService(dm_client, ls_client, mapping_service)

    existing_mapping = await sync_service.get_existing_dm_file_mapping(
        mapping.labeling_project_id
    )
    file_ids = list(existing_mapping.keys())
    if not file_ids:
        return StandardResponse(code="0", message="success", data=[])

    files_result = await db.execute(
        _select(DatasetFiles).where(DatasetFiles.id.in_(file_ids))
    )
    files = list(files_result.scalars().all())

    dataset_ids = {
        str(getattr(f, "dataset_id"))
        for f in files
        if getattr(f, "dataset_id", None)
    }
    dataset_name_map: dict[str, str] = {}
    if dataset_ids:
        ds_result = await db.execute(
            _select(Dataset.id, Dataset.name).where(Dataset.id.in_(dataset_ids))
        )
        for ds_id, ds_name in ds_result.fetchall():
            dataset_name_map[str(ds_id)] = ds_name or ""

    data: list[dict] = []
    for f in files:
        fid = str(getattr(f, "id"))
        ds_id = (
            str(getattr(f, "dataset_id"))
            if getattr(f, "dataset_id", None)
            else None
        )
        item: dict = {
            "id": fid,
            "datasetId": ds_id,
            "datasetName": dataset_name_map.get(ds_id or "", ""),
            "fileName": getattr(f, "file_name", ""),
            "fileSize": int(getattr(f, "file_size", 0) or 0),
            "filePath": getattr(f, "file_path", ""),
            "fileType": getattr(f, "file_type", None),
        }
        data.append(item)

    return StandardResponse(code="0", message="success", data=data)


@router.put("/{mapping_id}/files", response_model=StandardResponse[bool])
async def update_manual_mapping_files(
    mapping_id: str = Path(..., description="映射ID (mapping UUID)"),
    body: UpdateAutoAnnotationTaskFilesRequest = ...,  # 复用通用结构：datasetId + fileIds
    db: AsyncSession = Depends(get_db),
):
    """更新手动标注映射所关联的 DM 文件集合（仅追加，不删除已有任务）。

    语义约定：
    - 映射创建时所同步的文件集合视为“基础集合”，后续编辑不支持移除；
    - 本接口只会为新增的 fileIds 在 Label Studio 中创建任务；
    - 不会删除任何现有任务（delete_orphans=False）。
    """

    from sqlalchemy import select as _select  # 本地导入
    from typing import Set as _Set, Dict as _Dict
    from app.db.models.dataset_management import DatasetFiles

    mapping_service = DatasetMappingService(db)
    mapping = await mapping_service.get_mapping_by_uuid(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    dm_client = DatasetManagementService(db)
    ls_client = LabelStudioClient(
        base_url=settings.label_studio_base_url,
        token=settings.label_studio_user_token,
    )
    sync_service = SyncService(dm_client, ls_client, mapping_service)

    requested_ids = {str(fid) for fid in (body.file_ids or [])}
    if not requested_ids:
        # 不做任何变更，但认为成功
        return StandardResponse(code="0", message="success", data=True)

    existing_mapping = await sync_service.get_existing_dm_file_mapping(
        mapping.labeling_project_id
    )
    existing_ids = set(existing_mapping.keys())

    # 仅对新增文件创建任务
    new_ids = sorted(requested_ids - existing_ids)
    if not new_ids:
        return StandardResponse(code="0", message="success", data=True)

    stmt = (
        _select(DatasetFiles.dataset_id, DatasetFiles.id)
        .where(DatasetFiles.id.in_(new_ids))
    )
    result = await db.execute(stmt)
    rows = result.fetchall()

    grouped: _Dict[str, _Set[str]] = {}
    resolved_ids: _Set[str] = set()

    for ds_id, fid in rows:
        if not ds_id or not fid:
            continue
        fid_str = str(fid)
        grouped.setdefault(str(ds_id), set()).add(fid_str)
        resolved_ids.add(fid_str)

    unresolved_ids = {str(fid) for fid in new_ids} - resolved_ids
    if unresolved_ids:
        logger.warning(
            "Some file_ids could not be resolved to dataset_id when updating manual mapping files: %s",
            ",".join(sorted(unresolved_ids)),
        )
        grouped.setdefault(str(mapping.dataset_id), set()).update(unresolved_ids)

    # 为所有涉及到的额外数据集提前配置本地存储（与创建逻辑保持一致）
    try:
        for extra_ds_id in grouped.keys():
            if str(extra_ds_id) == str(mapping.dataset_id):
                continue

            extra_local_storage_path = (
                f"{settings.label_studio_local_document_root}/{extra_ds_id}"
            )
            extra_storage_result = await ls_client.create_local_storage(
                project_id=int(mapping.labeling_project_id),
                path=extra_local_storage_path,
                title=f"Dataset_BLOB_{extra_ds_id}",
                use_blob_urls=True,
                description=(
                    f"Local storage for dataset {extra_ds_id} "
                    "(multi-dataset manual project, edit)"
                ),
            )
            if not extra_storage_result:
                logger.warning(
                    "Failed to configure extra local storage for project %s (dataset %s) when updating manual mapping files",
                    mapping.labeling_project_id,
                    extra_ds_id,
                )
            else:
                logger.info(
                    "Extra local storage configured for project %s: %s (edit)",
                    mapping.labeling_project_id,
                    extra_local_storage_path,
                )
    except Exception as e:  # pragma: no cover
        logger.warning(
            "Error while configuring extra local storage for project %s during update: %s",
            mapping.labeling_project_id,
            e,
        )

    # 将新增文件按数据集分组，同步到 Label Studio；不删除已有任务
    for ds_id, ds_file_ids in grouped.items():
        await sync_service.sync_files(
            mapping,
            batch_size=100,
            allowed_file_ids=ds_file_ids,
            override_dataset_id=ds_id,
            delete_orphans=False,
        )

    return StandardResponse(code="0", message="success", data=True)


@router.post("/{mapping_id}/sync-label-studio-back", response_model=StandardResponse[bool])
async def import_manual_from_label_studio_to_dataset(
    mapping_id: str = Path(..., description="映射ID (mapping UUID)"),
    body: ImportFromLabelStudioRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """将手动标注工程在 Label Studio 中的标注结果导回到某个数据集。

    行为与自动标注的后向同步保持一致：
    - 按 mapping_id 定位 Label Studio 项目；
    - 通过 exportType 导出项目完整结果（JSON/COCO/YOLO 等）；
    - 将导出文件作为一个普通文件保存到指定数据集目录，并注册到 t_dm_dataset_files；
    - 不解析每条标注、不修改 tags，仅追加一个“导出工件”文件。
    """

    import os
    import tempfile
    from datetime import datetime

    if body is None:
        raise HTTPException(status_code=400, detail="Request body is required")

    # 1. 获取并校验映射（手动标注任务）
    mapping_service = DatasetMappingService(db)
    mapping = await mapping_service.get_mapping_by_uuid(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    # 2. 确定目标数据集：若未显式指定，则使用映射绑定的数据集
    dm_service = DatasetManagementService(db)
    target_dataset_id = body.target_dataset_id or mapping.dataset_id
    dataset = await dm_service.get_dataset(target_dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Target dataset not found")

    # 3. 调用 Label Studio 导出接口
    ls_client = LabelStudioClient(
        base_url=settings.label_studio_base_url,
        token=settings.label_studio_user_token,
    )

    export_format = (body.export_format or "JSON").upper()

    ext_map = {
        "JSON": ".json",
        "JSON_MIN": ".json",
        "CSV": ".csv",
        "TSV": ".tsv",
        "COCO": ".json",
        "YOLO": ".json",
        "YOLOV8": ".json",
    }
    file_ext = ext_map.get(export_format, ".json")

    project_id = mapping.labeling_project_id
    content = await ls_client.export_project(int(project_id), export_type=export_format)
    if content is None or len(content) == 0:
        raise HTTPException(status_code=500, detail="Failed to export project from Label Studio")

    # 4. 将导出结果写入临时文件，再通过 DatasetManagementService 导入到数据集
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    os.close(tmp_fd)

    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        def _sanitize_base_name(raw: str) -> str:
            # 去掉路径分隔符，仅保留最后一段
            name = (raw or "").strip().replace("\\", "/").split("/")[-1]
            # 去掉用户自带的扩展名，避免与服务器推断的后缀冲突
            if "." in name:
                name = name.rsplit(".", 1)[0]
            return name or f"ls_export_{project_id}_{timestamp}"

        if getattr(body, "file_name", None):
            base_stem = _sanitize_base_name(body.file_name)  # type: ignore[arg-type]
            base_name = f"{base_stem}{file_ext}"
        else:
            base_name = f"ls_export_{project_id}_{timestamp}{file_ext}"

        tmp_dir = os.path.dirname(tmp_path)
        target_tmp_path = os.path.join(tmp_dir, base_name)
        os.replace(tmp_path, target_tmp_path)

        # 写入目标（通常为源）数据集下的 "标注数据" 目录
        await dm_service.add_files_to_dataset_subdir(target_dataset_id, [target_tmp_path], "标注数据")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if "target_tmp_path" in locals() and os.path.exists(target_tmp_path):
            try:
                os.remove(target_tmp_path)
            except Exception:
                pass

    return StandardResponse(code="0", message="success", data=True)


@router.post("/{mapping_id}/sync-db", response_model=StandardResponse[int])
async def sync_manual_annotations_to_database(
    mapping_id: str = Path(..., description="映射ID (mapping UUID)"),
    db: AsyncSession = Depends(get_db),
):
    """从 Label Studio 项目同步当前手动标注结果到 DM 数据库。

        行为：
        - 基于 mapping_id 定位 Label Studio 项目；
        - 先执行 DM -> Label Studio 的文件差异同步（增量创建新文件任务、删除孤儿任务）；
        - 再执行双向标签同步（LS <-> DM），按时间戳保留较新标注：
            * LS 更新时间更新 -> 覆盖 DM；
            * DM tags_updated_at 更新 -> 覆盖 LS。
        返回值为同步变更数量（LS->DM + DM->LS）。
    """

    mapping_service = DatasetMappingService(db)
    mapping = await mapping_service.get_mapping_by_uuid(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    ls_client = LabelStudioClient(
        base_url=settings.label_studio_base_url,
        token=settings.label_studio_user_token,
    )

    dm_client = DatasetManagementService(db)
    sync_orchestrator = SyncService(dm_client, ls_client, mapping_service)
    file_sync_result = await sync_orchestrator.sync_files(mapping, batch_size=50)
    logger.info(
        "Manual sync-db pre file-sync done: mapping=%s, created=%s, deleted=%s, total=%s",
        mapping_id,
        file_sync_result.get("created", 0),
        file_sync_result.get("deleted", 0),
        file_sync_result.get("total", 0),
    )

    annotation_sync_result = await sync_orchestrator.sync_annotations_bidirectional(
        mapping,
        batch_size=50,
        overwrite=True,
        overwrite_ls=True,
        sync_files_first=False,
    )

    total_synced = annotation_sync_result.synced_to_dm + annotation_sync_result.synced_to_ls
    return StandardResponse(code="0", message="success", data=total_synced)

@router.get("", response_model=StandardResponse[PaginatedData[DatasetMappingResponse]])
async def list_mappings(
    page: int = Query(1, ge=1, description="页码（从1开始）"),
    size: int = Query(20, ge=1, le=100, description="每页记录数"),
    include_template: bool = Query(False, description="是否包含模板详情", alias="includeTemplate"),
    db: AsyncSession = Depends(get_db)
):
    """
    查询所有映射关系（分页）

    返回所有有效的数据集映射关系（未被软删除的），支持分页查询。
    可选择是否包含完整的标注模板信息（默认不包含，以提高列表查询性能）。

    参数:
    - page: 页码（从1开始）
    - pageSize: 每页记录数
    - includeTemplate: 是否包含模板详情（默认false）
    """
    try:
        service = DatasetMappingService(db)

        # 计算 skip
        skip = (page - 1) * size

        logger.info(f"List mappings: page={page}, size={size}, include_template={include_template}")

        # 获取数据和总数
        mappings, total = await service.get_all_mappings_with_count(
            skip=skip,
            limit=size,
            include_deleted=False,
            include_template=include_template
        )

        # 计算总页数
        total_pages = math.ceil(total / size) if total > 0 else 0

        # 构造分页响应
        paginated_data = PaginatedData(
            page=page,
            size=size,
            total_elements=total,
            total_pages=total_pages,
            content=mappings
        )

        logger.info(f"List mappings: page={page}, returned {len(mappings)}/{total}, templates_included: {include_template}")

        return StandardResponse(
            code="0",
            message="success",
            data=paginated_data
        )

    except Exception as e:
        logger.error(f"Error listing mappings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{mapping_id}", response_model=StandardResponse[DatasetMappingResponse])
async def get_mapping(
    mapping_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    根据 UUID 查询单个映射关系（包含关联的标注模板详情）

    返回数据集映射关系以及关联的完整标注模板信息，包括：
    - 映射基本信息
    - 数据集信息
    - Label Studio 项目信息
    - 完整的标注模板配置（如果存在）
    """
    try:
        service = DatasetMappingService(db)

        logger.info(f"Get mapping with template details: {mapping_id}")

        # 获取映射，并包含完整的模板信息
        mapping = await service.get_mapping_by_uuid(mapping_id, include_template=True)

        if not mapping:
            raise HTTPException(
                status_code=404,
                detail=f"Mapping not found: {mapping_id}"
            )

        logger.info(f"Found mapping: {mapping.id}, template_included: {mapping.template is not None}")

        return StandardResponse(
            code="0",
            message="success",
            data=mapping
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mapping: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/by-source/{dataset_id}", response_model=StandardResponse[PaginatedData[DatasetMappingResponse]])
async def get_mappings_by_source(
    dataset_id: str,
    page: int = Query(1, ge=1, description="页码（从1开始）"),
    size: int = Query(20, ge=1, le=100, description="每页记录数"),
    include_template: bool = Query(True, description="是否包含模板详情", alias="includeTemplate"),
    db: AsyncSession = Depends(get_db)
):
    """
    根据源数据集 ID 查询所有映射关系（分页，包含模板详情）

    返回该数据集创建的所有标注项目（不包括已删除的），支持分页查询。
    默认包含关联的完整标注模板信息。

    参数:
    - dataset_id: 数据集ID
    - page: 页码（从1开始）
    - pageSize: 每页记录数
    - includeTemplate: 是否包含模板详情（默认true）
    """
    try:
        service = DatasetMappingService(db)

        # 计算 skip
        skip = (page - 1) * size

        logger.info(f"Get mappings by source dataset id: {dataset_id}, page={page}, size={size}, include_template={include_template}")

        # 获取数据和总数（包含模板信息）
        mappings, total = await service.get_mappings_by_source_with_count(
            dataset_id=dataset_id,
            skip=skip,
            limit=size,
            include_template=include_template
        )

        # 计算总页数
        total_pages = math.ceil(total / size) if total > 0 else 0

        # 构造分页响应
        paginated_data = PaginatedData(
            page=page,
            size=size,
            total_elements=total,
            total_pages=total_pages,
            content=mappings
        )

        logger.info(f"Found {len(mappings)} mappings on page {page}, total: {total}, templates_included: {include_template}")

        return StandardResponse(
            code="0",
            message="success",
            data=paginated_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mappings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{project_id}", response_model=StandardResponse[DeleteDatasetResponse])
async def delete_mapping(
    project_id: str = Path(..., description="映射UUID（path param）"),
    db: AsyncSession = Depends(get_db)
):
    """
    删除映射关系和对应的 Label Studio 项目

    通过 path 参数 `project_id` 指定要删除的映射（映射的 UUID）。

    此操作会：
    1. 删除 Label Studio 中的项目
    2. 软删除数据库中的映射记录
    """
    try:
        logger.debug(f"Delete mapping request received: project_id={project_id!r}")

        ls_client = LabelStudioClient(base_url=settings.label_studio_base_url,
                                      token=settings.label_studio_user_token)
        service = DatasetMappingService(db)

        # 使用 mapping UUID 查询映射记录
        logger.debug(f"Deleting by mapping UUID: {project_id}")
        mapping = await service.get_mapping_by_uuid(project_id)

        logger.debug(f"Mapping lookup result: {mapping}")

        if not mapping:
            raise HTTPException(
                status_code=404,
                detail=f"Mapping either not found or not specified."
            )

        id = mapping.id
        labeling_project_id = mapping.labeling_project_id

        logger.debug(f"Found mapping: {id}, Label Studio project ID: {labeling_project_id}")

        # 1. 删除 Label Studio 项目
        try:
            logger.debug(f"Deleting Label Studio project: {labeling_project_id}")
            delete_success = await ls_client.delete_project(int(labeling_project_id))
            if delete_success:
                logger.debug(f"Successfully deleted Label Studio project: {labeling_project_id}")
            else:
                logger.warning(f"Failed to delete Label Studio project or project not found: {labeling_project_id}")
        except Exception as e:
            logger.error(f"Error deleting Label Studio project: {e}")
            # 继续执行，即使 Label Studio 项目删除失败也要删除映射记录

        # 2. 软删除映射记录
        soft_delete_success = await service.soft_delete_mapping(id)
        logger.debug(f"Soft delete result for mapping {id}: {soft_delete_success}")

        if not soft_delete_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete mapping record"
            )

        logger.info(f"Successfully deleted mapping: {id}, Label Studio project: {labeling_project_id}")

        return StandardResponse(
            code="0",
            message="success",
            data=DeleteDatasetResponse(
                id=id,
                status="success"
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting mapping: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
