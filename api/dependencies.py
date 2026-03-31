"""FastAPI dependency injection for shared services."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.consulting_service import ConsultingService


@lru_cache(maxsize=1)
def get_consulting_service() -> ConsultingService:
    """Return the singleton ConsultingService (initialised once on first call)."""
    base = Path(__file__).parent.parent
    db_path = str(base / "data" / "aic.db")
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "memory").mkdir(exist_ok=True)
    return ConsultingService(memory_dir=str(base / "memory"), db_path=db_path)
