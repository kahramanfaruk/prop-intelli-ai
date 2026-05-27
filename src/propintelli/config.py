"""Centralised, environment-driven configuration.

All runtime configuration is resolved here through :class:`Settings`, a
``pydantic-settings`` model populated from environment variables and an optional
``.env`` file. Every field carries a safe default so the pipeline runs fully
offline without any configuration. Secrets (API keys) are never hard-coded; they
are read from the environment only.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmProvider(enum.StrEnum):
    """Selectable Layer-B extraction backends.

    Notes
    -----
    ``NONE`` keeps the pipeline deterministic and offline (the default). The
    remaining members enable an LLM second opinion that is reconciled with the
    deterministic Layer-A result.
    """

    NONE = "none"
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class PromptVariant(enum.StrEnum):
    """Documented prompt-engineering variants for the LLM layer."""

    V1_DIRECT = "v1_direct"
    V2_SCHEMA = "v2_schema"
    V3_REASONING = "v3_reasoning"


class Settings(BaseSettings):
    """Application settings resolved from the environment and ``.env``.

    Attributes
    ----------
    data_dir : Path
        Root directory for the medallion layers (Bronze raw store, Silver SQLite
        database, Gold DuckDB and file exports).
    llm_provider : LlmProvider
        Backend used by the optional LLM extraction layer.
    llm_prompt_variant : PromptVariant
        Prompt template the LLM layer sends.
    confidence_auto_approve : float
        Records at or above this overall confidence are auto-approved.
    confidence_review_floor : float
        Records below this confidence require manual correction; values between
        ``review_floor`` and ``auto_approve`` are flagged for review.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROPINTELLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Storage / data layout -------------------------------------------------
    data_dir: Path = Field(default=Path("./data"))

    # Extraction: Layer B (LLM) -------------------------------------------
    llm_provider: LlmProvider = LlmProvider.NONE
    llm_prompt_variant: PromptVariant = PromptVariant.V2_SCHEMA
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PROPINTELLI_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-4o-mini"

    azure_openai_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PROPINTELLI_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
    )
    azure_openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PROPINTELLI_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )
    azure_openai_deployment: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PROPINTELLI_AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT"
        ),
    )
    azure_openai_api_version: str = "2024-06-01"

    # Preprocessing / OCR ---------------------------------------------------
    ocr_enabled: bool = False
    ocr_language: str = "deu"
    scanned_text_threshold: int = Field(default=50, ge=0)

    # Confidence-driven human-in-the-loop routing ---------------------------
    confidence_auto_approve: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_review_floor: float = Field(default=0.60, ge=0.0, le=1.0)

    # Logging ---------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Normalise and validate the log level.

        Parameters
        ----------
        value : str
            Raw log-level string from the environment.

        Returns
        -------
        str
            Upper-cased log level.

        Raises
        ------
        ValueError
            If the level is not a recognised ``logging`` level name.
        """
        normalised = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalised not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return normalised

    def model_post_init(self, _context: object) -> None:
        """Validate cross-field invariants after construction.

        Raises
        ------
        ValueError
            If the review floor is not strictly below the auto-approve
            threshold, which would make the ``needs_review`` band empty or
            inverted.
        """
        if self.confidence_review_floor >= self.confidence_auto_approve:
            msg = (
                "confidence_review_floor "
                f"({self.confidence_review_floor}) must be < "
                f"confidence_auto_approve ({self.confidence_auto_approve})"
            )
            raise ValueError(msg)

    # Derived medallion-layer paths ----------------------------------------
    @property
    def bronze_dir(self) -> Path:
        """Path to the Bronze raw-document store."""
        return self.data_dir / "bronze"

    @property
    def silver_db_path(self) -> Path:
        """Path to the Silver SQLite database file."""
        return self.data_dir / "silver" / "propintelli.sqlite"

    @property
    def gold_dir(self) -> Path:
        """Path to the Gold analytics export directory."""
        return self.data_dir / "gold"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns
    -------
    Settings
        The cached settings instance, constructed on first access from the
        environment and ``.env`` file.
    """
    return Settings()
