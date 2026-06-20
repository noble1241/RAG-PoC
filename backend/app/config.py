from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

# Resolve .env relative to this file's directory (backend/), not the CWD
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

    # Models
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # RAG tuning
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Server
    allowed_origins: list[str] = ["http://localhost:5173"]
    log_level: str = "INFO"

    @field_validator("openai_api_key")
    @classmethod
    def api_key_must_be_set(cls, v: str) -> str:
        if not v or v.startswith("sk-your"):
            raise ValueError("OPENAI_API_KEY must be set to a real key")
        return v


settings = Settings()
