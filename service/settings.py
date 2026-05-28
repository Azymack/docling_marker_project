from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SERVICE_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 50
    log_level: str = "info"
