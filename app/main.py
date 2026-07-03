"""
app/main.py
────────────
FastAPI application factory for the Agentic RAG — Lecturer Notes Updater.

Startup order
─────────────
1. Load settings (validates .env)
2. Register global exception handlers
3. Mount API routers
4. Add CORS middleware
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.agentic_rag import router as agentic_rag_router
from app.api.routes.export_notes import router as export_router
from app.api.routes.prompt import router as prompt_router
from app.api.routes.quota import router as quota_router
from app.api.routes.upload import router as upload_router
from app.core.config import get_settings
from app.core.exceptions import (
    AgenticRagError,
    DocumentParseError,
    EmbeddingError,
    ExportError,
    FileTooLargeError,
    PromptProcessingError,
    QuotaExhaustedError,
    RagAgentError,
    RateLimitExceededError,
    UnsupportedFileTypeError,
    VectorStoreError,
    WebResearchError,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks."""
    settings = get_settings()

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory ready: %s", settings.upload_dir)

    # Warm up Pinecone connection (non-fatal if credentials missing in dev)
    try:
        from app.services.vector_store import health_check
        if health_check():
            logger.info("Pinecone connection verified.")
        else:
            logger.warning(
                "Pinecone is not reachable. Set PINECONE_API_KEY in .env."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pinecone warm-up skipped: %s", exc)

    logger.info("%s v%s started.", settings.app_name, settings.app_version)
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Application shutting down.")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Agentic RAG web agent that helps lecturers update outdated course notes. "
            "FR1: Upload & Storage — accepts TXT / PDF / DOCX files, parses them, "
            "embeds the content with Gemini, and stores vectors in Pinecone. "
            "FR2: Prompt/Request Input — accepts natural-language update requests, "
            "retrieves relevant chunks from the vector store, and generates updated "
            "content with Gemini. "
            "FR3: Agentic RAG Processing — autonomous system that retrieves chunks, "
            "performs real-time web research, cross-checks information, synthesises "
            "verified content, delivers old-vs-new comparisons, applies self-correction "
            "(length/style/research quality), and supports iterative refinement."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ──────────────────────────────────────────────

    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_file_handler(req: Request, exc: UnsupportedFileTypeError):
        return JSONResponse(
            status_code=415,
            content={"status": "error", "error_type": "UnsupportedFileType", "message": str(exc)},
        )

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(req: Request, exc: FileTooLargeError):
        return JSONResponse(
            status_code=413,
            content={"status": "error", "error_type": "FileTooLarge", "message": str(exc)},
        )

    @app.exception_handler(DocumentParseError)
    async def parse_error_handler(req: Request, exc: DocumentParseError):
        return JSONResponse(
            status_code=422,
            content={"status": "error", "error_type": "DocumentParseError", "message": str(exc)},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(req: Request, exc: EmbeddingError):
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error_type": "EmbeddingError", "message": str(exc)},
        )

    @app.exception_handler(VectorStoreError)
    async def vector_store_error_handler(req: Request, exc: VectorStoreError):
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error_type": "VectorStoreError", "message": str(exc)},
        )

    @app.exception_handler(PromptProcessingError)
    async def prompt_processing_error_handler(req: Request, exc: PromptProcessingError):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": "PromptProcessingError", "message": str(exc)},
        )

    @app.exception_handler(AgenticRagError)
    async def agentic_rag_error_handler(req: Request, exc: AgenticRagError):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": "AgenticRagError", "message": str(exc)},
        )

    @app.exception_handler(WebResearchError)
    async def web_research_error_handler(req: Request, exc: WebResearchError):
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error_type": "WebResearchError", "message": str(exc)},
        )

    @app.exception_handler(ExportError)
    async def export_error_handler(req: Request, exc: ExportError):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": "ExportError", "message": str(exc)},
        )

    @app.exception_handler(QuotaExhaustedError)
    async def quota_exhausted_handler(req: Request, exc: QuotaExhaustedError):
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error_type": "QuotaExhausted",
                "message": exc.detail,
                "retry_after_seconds": exc.retry_after_seconds,
                "quota_exhausted": True,
            },
        )

    @app.exception_handler(RagAgentError)
    async def rag_agent_error_handler(req: Request, exc: RagAgentError):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_type": "RagAgentError", "message": str(exc)},
        )

    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_handler(req: Request, exc: RateLimitExceededError):
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error_type": "RateLimitExceeded",
                "message": exc.detail,
                "retry_after_seconds": exc.retry_after_seconds,
            },
        )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(upload_router)
    app.include_router(prompt_router)
    app.include_router(agentic_rag_router)
    app.include_router(export_router)
    app.include_router(quota_router)

    # ── Root endpoint ──────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "status": "running",
        }

    return app


# Module-level app instance used by uvicorn
app = create_app()