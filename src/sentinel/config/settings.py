"""Environment-based settings for Sentinel."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Central settings. Secrets are read from the environment, never committed."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="SENTINEL_",
        extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    data_dir: Path = Path("data/generated")
    database_url: str = "postgresql+psycopg://sentinel:change-me@localhost:5432/sentinel_db"
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout_seconds: float = 30
    llm_max_retries: int = 2
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64
    documents_dir: Path = Path("data/documents")
    rag_top_k: int = 5
    rag_min_similarity: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
