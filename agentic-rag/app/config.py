""" Centralized settings — loaded once, injected everywhere."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str
    claude_model: str = "claude-3-5-sonnet-20241022"

    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection: str = "agentic_rag"

    memory_db_path: str = "./storage/memory.sqlite"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    retrieval_k: int = 4
    grade_threshold: float = 0.6
    max_search_results: int = 5
    max_rewrite_attempts: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Ensure storage directories exist
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    return settings
