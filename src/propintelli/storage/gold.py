"""Gold layer — analytics-ready exports built with DuckDB.

The Gold layer flattens validated Silver records into a wide analytics table and
a long features table, persists them as a DuckDB database plus Parquet and CSV
exports, and computes a city-level market summary. DuckDB and Parquet map onto a
Microsoft Fabric Lakehouse / OneLake in production; the columnar exports are the
hand-off point for BI tools such as Power BI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from propintelli.logging_setup import get_logger
from propintelli.schemas.property_record import Features, PropertyRecord

logger = get_logger(__name__)

_DB_NAME = "analytics.duckdb"
_FEATURE_NAMES = tuple(Features.model_fields)

# (column name, DuckDB type) for the wide analytics table, in insertion order.
_PROPERTY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("property_id", "VARCHAR"),
    ("source_document", "VARCHAR"),
    ("listing_type", "VARCHAR"),
    ("price_kind", "VARCHAR"),
    ("price_eur", "DOUBLE"),
    ("price_per_sqm", "DOUBLE"),
    ("living_area_sqm", "DOUBLE"),
    ("plot_area_sqm", "DOUBLE"),
    ("rooms", "DOUBLE"),
    ("floor", "INTEGER"),
    ("total_floors", "INTEGER"),
    ("year_built", "INTEGER"),
    ("condition", "VARCHAR"),
    ("city", "VARCHAR"),
    ("postal_code", "VARCHAR"),
    ("district", "VARCHAR"),
    ("energy_class", "VARCHAR"),
    ("heating_type", "VARCHAR"),
    ("energy_demand_kwh", "DOUBLE"),
    ("availability_date", "DATE"),
    ("overall_confidence", "DOUBLE"),
    ("completeness", "DOUBLE"),
    ("validation_pass_rate", "DOUBLE"),
    ("review_status", "VARCHAR"),
    ("extracted_at", "TIMESTAMP"),
)


@dataclass(frozen=True, slots=True)
class GoldArtifacts:
    """Paths and summary produced by a Gold build.

    Attributes
    ----------
    duckdb_path : Path
        The persisted DuckDB analytics database.
    properties_parquet, properties_csv : Path
        The wide analytics table exports.
    features_parquet : Path
        The long features table export.
    summary_csv : Path
        The city-level market summary export.
    summary : list of dict
        The city-level summary rows.
    """

    duckdb_path: Path
    properties_parquet: Path
    properties_csv: Path
    features_parquet: Path
    summary_csv: Path
    summary: list[dict[str, Any]]


def _property_row(record: PropertyRecord) -> tuple[Any, ...]:
    """Flatten a record into a row matching ``_PROPERTY_COLUMNS``."""
    return (
        record.property_id,
        record.source_document,
        record.listing_type.value if record.listing_type else None,
        record.price_kind.value if record.price_kind else None,
        float(record.price_eur) if record.price_eur is not None else None,
        float(record.price_per_sqm) if record.price_per_sqm is not None else None,
        record.living_area_sqm,
        record.plot_area_sqm,
        record.rooms,
        record.floor,
        record.total_floors,
        record.year_built,
        record.condition.value if record.condition else None,
        record.location.city,
        record.location.postal_code,
        record.location.district,
        record.energy.energy_class.value if record.energy.energy_class else None,
        record.energy.heating_type.value if record.energy.heating_type else None,
        record.energy.energy_demand_kwh,
        record.availability_date,
        record.quality.overall_confidence,
        record.quality.completeness,
        record.quality.validation_pass_rate,
        record.quality.review_status.value,
        record.extracted_at,
    )


def _feature_rows(record: PropertyRecord) -> list[tuple[str, str, bool]]:
    """Flatten a record's set features into ``(property_id, name, value)`` rows."""
    rows: list[tuple[str, str, bool]] = []
    for name in _FEATURE_NAMES:
        value = getattr(record.features, name)
        if value is not None:
            rows.append((record.property_id, name, bool(value)))
    return rows


def build_gold(records: list[PropertyRecord], gold_dir: Path) -> GoldArtifacts:
    """Build the Gold analytics database, exports, and market summary.

    Parameters
    ----------
    records : list of PropertyRecord
        The validated Silver records to publish.
    gold_dir : Path
        Output directory for the DuckDB database and file exports.

    Returns
    -------
    GoldArtifacts
        Paths to the generated artifacts and the computed summary rows.
    """
    gold_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = gold_dir / _DB_NAME
    properties_parquet = gold_dir / "properties.parquet"
    properties_csv = gold_dir / "properties.csv"
    features_parquet = gold_dir / "features.parquet"
    summary_csv = gold_dir / "market_summary.csv"

    if duckdb_path.exists():
        duckdb_path.unlink()

    connection = duckdb.connect(str(duckdb_path))
    try:
        _create_tables(connection)
        _insert_rows(connection, records)
        for table, path in (
            ("properties", properties_parquet),
            ("features", features_parquet),
        ):
            connection.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
        connection.execute(f"COPY properties TO '{properties_csv}' (HEADER, DELIMITER ',')")
        summary = _market_summary(connection)
        connection.execute(
            "COPY (SELECT * FROM market_summary) TO ? (HEADER, DELIMITER ',')",
            [str(summary_csv)],
        )
    finally:
        connection.close()

    logger.info("gold_built", extra={"records": len(records), "gold_dir": str(gold_dir)})
    return GoldArtifacts(
        duckdb_path=duckdb_path,
        properties_parquet=properties_parquet,
        properties_csv=properties_csv,
        features_parquet=features_parquet,
        summary_csv=summary_csv,
        summary=summary,
    )


def _create_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the wide properties table and the long features table."""
    columns = ", ".join(f"{name} {sql_type}" for name, sql_type in _PROPERTY_COLUMNS)
    connection.execute(f"CREATE TABLE properties ({columns})")
    connection.execute(
        "CREATE TABLE features (property_id VARCHAR, feature_name VARCHAR, value BOOLEAN)"
    )


def _insert_rows(connection: duckdb.DuckDBPyConnection, records: list[PropertyRecord]) -> None:
    """Insert property and feature rows for every record."""
    if not records:
        return
    placeholders = ", ".join("?" for _ in _PROPERTY_COLUMNS)
    connection.executemany(
        f"INSERT INTO properties VALUES ({placeholders})",
        [_property_row(record) for record in records],
    )
    feature_rows = [row for record in records for row in _feature_rows(record)]
    if feature_rows:
        connection.executemany("INSERT INTO features VALUES (?, ?, ?)", feature_rows)


def _market_summary(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Build and return a city-level market summary for sale listings."""
    connection.execute(
        """
        CREATE TABLE market_summary AS
        SELECT
            city,
            count(*) AS listings,
            round(avg(price_per_sqm), 2) AS avg_price_per_sqm,
            round(avg(living_area_sqm), 1) AS avg_living_area_sqm,
            round(avg(overall_confidence), 3) AS avg_confidence
        FROM properties
        WHERE listing_type = 'sale' AND city IS NOT NULL
        GROUP BY city
        ORDER BY listings DESC, city
        """
    )
    cursor = connection.execute("SELECT * FROM market_summary")
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
