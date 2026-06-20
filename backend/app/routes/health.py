from fastapi import APIRouter
from app.schemas import HealthResponse, ReadyResponse
from app.vectorstore import ping_chroma

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    chroma_ok = await ping_chroma()
    return ReadyResponse(
        status="ok" if chroma_ok else "degraded",
        chroma="ok" if chroma_ok else "unreachable",
    )
