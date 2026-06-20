from __future__ import annotations

from pydantic import BaseModel, field_validator


class IngestTextRequest(BaseModel):
    text: str
    source: str = "paste"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class IngestResponse(BaseModel):
    document_id: str
    source: str
    chunk_count: int
    tokens_processed: int


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        if len(v) > 4096:
            raise ValueError("query must be at most 4096 characters")
        return v


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    content: str
    score: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    chroma: str
