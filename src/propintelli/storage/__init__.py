"""Storage: medallion Silver (SQLAlchemy) and Gold (DuckDB/Parquet)."""

from __future__ import annotations

from propintelli.storage.gold import GoldArtifacts, build_gold
from propintelli.storage.repository import ProcessingRunInfo, SilverRepository

__all__ = ["GoldArtifacts", "ProcessingRunInfo", "SilverRepository", "build_gold"]
