"""Application configuration using Pydantic Settings.

Reads from environment variables or .env files, providing type safety
and sensible defaults for local development and Kubernetes deployments.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema."""

    # Project metadata
    PROJECT_NAME: str = "Real-Time Transaction Fraud Detection API"
    PROJECT_DESCRIPTION: str = (
        "Production-grade sub-millisecond fraud scoring microservice "
        "with automated MLOps pipelines and Google Cloud Platform (GCP) GKE deployment."
    )
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["*"]

    # Model and artifacts
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: str = "models/fraud_model_pipeline.joblib"
    METADATA_PATH: str = "models/model_metadata.json"
    FALLBACK_THRESHOLD: float = 0.8270

    # Observability
    ENABLE_METRICS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

