"""
scripts/recreate_pinecone_index.py
────────────────────────────────────
Standalone initialisation script to delete and re-create the
Pinecone index with the dimension and metric from settings.

Usage:
    python scripts/recreate_pinecone_index.py

What it does:
    1. Loads settings from .env (PINECONE_API_KEY, index name, etc.)
    2. Deletes the existing Pinecone index (if it exists)
    3. Re-creates it with dimension=3072 and metric='cosine'
    4. Confirms the new index is ready

Run this ONCE after switching to a new embedding model that outputs
a different vector dimension (e.g. gemini-embedding-001 → 3072).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import app modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        from app.services.vector_store import recreate_index

        logger.info("Starting Pinecone index recreation …")
        recreate_index()
        logger.info("Pinecone index recreation complete.")
    except Exception as exc:
        logger.exception("Pinecone index recreation FAILED: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()