"""Structured logging configuration.

Provides a single :func:`configure_logging` entry point that installs either a
human-readable console handler or a JSON handler (selected via settings). JSON
logs map cleanly onto Azure Application Insights / Azure Monitor in production,
while the console format keeps local development readable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from propintelli.config import Settings, get_settings

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Attributes present on every ``LogRecord``; anything else is treated as
# structured context and emitted alongside the message in JSON mode.
_RESERVED_RECORD_KEYS = frozenset(
    logging.makeLogRecord({}).__dict__.keys() | {"message", "asctime", "taskName"}
)


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Extra keyword arguments passed to a logging call (via ``extra=...``) are
    merged into the emitted object, enabling structured, queryable logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The record to serialise.

        Returns
        -------
        str
            A single-line JSON document describing the record.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings | None = None) -> None:
    """Install the root logging configuration.

    Idempotent: repeated calls replace the existing handlers rather than
    stacking duplicates. Safe to call at every process entry point (CLI, UI,
    worker).

    Parameters
    ----------
    settings : Settings or None, optional
        Settings to read the log level and format from. Defaults to the
        process-wide settings singleton.
    """
    settings = settings or get_settings()
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Parameters
    ----------
    name : str
        Logger name, conventionally the calling module's ``__name__``.

    Returns
    -------
    logging.Logger
        A logger that inherits the root configuration.
    """
    return logging.getLogger(name)
