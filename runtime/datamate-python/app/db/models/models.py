from sqlalchemy import Boolean, Column, String, TIMESTAMP

from app.db.models.base_entity import BaseEntity


class Models(BaseEntity):
    """模型配置表，对应表 t_models。普通用户按创建者隔离，管理员可跨用户管理。

    CREATE TABLE IF NOT EXISTS t_models (
        id         VARCHAR(36) PRIMARY KEY COMMENT '主键ID',
        model_name VARCHAR(100) NOT NULL COMMENT '模型名称（如 qwen2）',
        provider   VARCHAR(50)  NOT NULL COMMENT '模型提供商（如 Ollama、OpenAI、DeepSeek）',
        base_url   VARCHAR(255) NOT NULL COMMENT 'API 基础地址',
        api_key    VARCHAR(512) DEFAULT '' COMMENT 'API 密钥（无密钥则为空）',
        type       VARCHAR(50)  NOT NULL COMMENT '模型类型（如 chat、embedding）',
        is_enabled TINYINT      DEFAULT 1 COMMENT '是否启用：1-启用，0-禁用',
        is_default TINYINT      DEFAULT 0 COMMENT '是否默认：1-默认，0-非默认',
        created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        updated_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
        created_by VARCHAR(255) COMMENT '创建者',
        updated_by VARCHAR(255) COMMENT '更新者',
        UNIQUE KEY uk_model_provider (model_name, provider)
    ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT ='模型配置表';
    """

    __tablename__ = "t_models"
    __admin_ignore_data_scope__ = True

    id = Column(String(36), primary_key=True, index=True, comment="主键ID")
    model_name = Column(String(100), nullable=False, comment="模型名称（如 qwen2）")
    provider = Column(String(50), nullable=False, comment="模型提供商（如 Ollama、OpenAI、DeepSeek）")
    base_url = Column(String(255), nullable=False, comment="API 基础地址")
    api_key = Column(String(512), nullable=False, default="", comment="API 密钥（无密钥则为空）")
    type = Column(String(50), nullable=False, comment="模型类型（如 chat、embedding）")

    is_enabled = Column(Boolean, nullable=False, default=True, comment="是否启用")
    is_default = Column(Boolean, nullable=False, default=False, comment="是否默认")
    is_deleted = Column(Boolean, nullable=False, default=False, comment="是否删除")

    __table_args__ = (
        # 与 DDL 中的 uk_model_provider 保持一致
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "模型配置表",
        },
    )
