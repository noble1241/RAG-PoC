from __future__ import annotations

import logging
import logging.config
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.routes import chat, documents, health

# ── Structured JSON logging ────────────────────────────────────────────────────
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "logging.Formatter",
                "fmt": '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":%(message)s}',
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {"handlers": ["stdout"], "level": settings.log_level},
    }
)

logger = logging.getLogger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Service", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Correlation ID middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    import time

    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        '"method":"%s","path":"%s","status":%d,"latency_ms":%s,"request_id":"%s"',
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["ops"])
app.include_router(documents.router, tags=["ingestion"])
app.include_router(chat.router, tags=["chat"])
