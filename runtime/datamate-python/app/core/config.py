from pydantic_settings import BaseSettings
from pydantic import model_validator
from sqlalchemy import URL
from typing import Optional

class Settings(BaseSettings):
    """应用程序配置"""

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'

    # Service
    app_name: str = "DataMate Python Backend"
    app_version: str = "1.0.0"
    app_description: str = "Adapter for integrating Data Management System with Label Studio"

    host: str = "0.0.0.0"
    port: int = 18000

    # CORS
    # allowed_origins: List[str] = ["*"]
    # allowed_methods: List[str] = ["*"]
    # allowed_headers: List[str] = ["*"]

    # Log
    log_level: str = "INFO"
    debug: bool = True
    log_file_dir: str = "/var/log/datamate/backend-python"
    log_pvc_monitor_enabled: bool = True
    log_pvc_monitor_path: str = "/var/log/datamate"
    log_pvc_monitor_threshold: float = 0.9
    log_pvc_monitor_interval_seconds: int = 300
    log_pvc_monitor_delete_batch_size: int = 10
    log_pvc_monitor_file_suffixes: str = "log,out,err"
    log_rotation_max_size: str = "100MB"
    log_rotation_backup_count: int = 30
    rag_storage_dir: str = "/data/rag_storage"

    # Database
    pgsql_host: str = "datamate-database"
    pgsql_port: int = 5432
    pgsql_user: str = "postgres"
    pgsql_password: str = ""
    pgsql_database: str = "datamate"

    # Database
    mysql_host: str = "datamate-database"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "datamate"

    database_url: str = ""  # Will be overridden by build_database_url() if not provided

    @model_validator(mode='after')
    def build_database_url(self):
        """如果没有提供 database_url，则根据数据库配置构建。"""
        if not self.database_url:
            if self.pgsql_host:
                if self.pgsql_password and self.pgsql_user:
                    self.database_url = URL.create(
                        "postgresql+asyncpg",
                        username=self.pgsql_user,
                        password=self.pgsql_password,
                        host=self.pgsql_host,
                        port=self.pgsql_port,
                        database=self.pgsql_database,
                    ).render_as_string(hide_password=False)
                else:
                    self.database_url = f"postgresql+asyncpg://{self.pgsql_host}:{self.pgsql_port}/{self.pgsql_database}"
            elif self.mysql_password and self.mysql_user:
                self.database_url = URL.create(
                    "mysql+aiomysql",
                    username=self.mysql_user,
                    password=self.mysql_password,
                    host=self.mysql_host,
                    port=self.mysql_port,
                    database=self.mysql_database,
                ).render_as_string(hide_password=False)
            else:
                self.database_url = f"mysql+aiomysql://{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        return self


    # Label Studio
    label_studio_base_url: str = "http://label-studio:8000"
    label_studio_username: Optional[str] = None
    label_studio_password: Optional[str] = None
    label_studio_user_token: Optional[str] = None  # Legacy Token

    label_studio_local_document_root: str = "/label-studio/local"  # Label Studio local file storage path
    label_studio_file_path_prefix: str = "/data/local-files/?d="  # Label Studio local file serving URL prefix

    ls_task_page_size: int = 1000

    # DataMate
    dm_file_path_prefix: str = "/dataset"  # DM存储文件夹前缀

    datamate_jwt_enable: bool = False

    # Milvus 配置
    milvus_uri: str = "http://milvus:19530"
    milvus_token: str = ""

    # 文件存储配置（共享文件系统）
    file_storage_path: str = "/data/files"

    # ==================== 配比任务并行复制配置 ====================
    # 动态并发计算参数（全闪存储高性能场景默认值）
    
    # 并发下限（最少并发数）
    ratio_copy_min_concurrent: int = 8
    
    # 并发上限（最多并发数，防止资源耗尽）
    ratio_copy_max_concurrent: int = 128
    
    # CPU核心系数（每个核心贡献的并发数，全闪存储建议4.0）
    ratio_copy_cpu_factor: float = 4.0
    
    # 每并发任务预估内存占用（MB）
    ratio_copy_memory_per_task_mb: int = 32
    
    # 内存安全保留比例（保留给其他进程）
    ratio_copy_memory_reserve_ratio: float = 0.2
    
    # 是否启用动态计算（False则使用固定值）
    ratio_copy_dynamic_concurrent: bool = True
    
    # 固定并发数（当 dynamic_concurrent=False 时使用）
    ratio_copy_fixed_concurrent: int = 10

# 全局设置实例
settings = Settings()
