"""
app/core/config.py
──────────────────
Centralised, validated settings loaded from the .env file.
All downstream modules import from here — never read os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Google Gemini (used for embeddings) ──────────
    google_api_key: str = ""

    # ── OpenRouter (used for generation) ─────────────
    openrouter_api_key: str = ""

    # ── Pinecone ───────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "lecturer-notes"
    pinecone_environment: str = "us-east-1"

    # ── App ────────────────────────────────────────
    app_name: str = "Agentic RAG — Lecturer Notes Updater"
    app_version: str = "1.0.0"
    debug: bool = False
    upload_dir: str = "uploads"
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf,txt,docx"

    # ── Embeddings ─────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 3072
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── LLM Generation ───────────────────────────────
    # Gemini model used for embeddings (FR2 prompt service)
    embedding_generation_model: str = "gemini-1.5-flash"
    # OpenRouter model slug used for agentic RAG generation (FR3)
    openrouter_generation_model: str = "openai/gpt-4o-mini"
    generation_temperature: float = 0.3
    generation_max_output_tokens: int = 4096

    # ── Web Research (FR3) ─────────────────────────
    web_search_provider: str = "duckduckgo"
    max_research_sources: int = 5
    academic_domains: str = (
        "edu,semanticscholar.org,arxiv.org,scholar.google.com,"
        "wikipedia.org,nature.com,sciencedirect.com,springer.com,"
        "ieee.org,acm.org"
    )

    # ── Self-Correction (FR3) ──────────────────────
    max_length_deviation_ratio: float = 0.4
    min_quality_score: float = 0.6
    max_self_correction_rounds: int = 2

    # ── Session (FR3) ──────────────────────────────
    session_max_iterations: int = 10
    session_ttl_minutes: int = 120

    # ── Derived helpers ────────────────────────────
    @property
    def allowed_ext_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton — safe to call anywhere."""
    return Settings()