"""The no-op extraction provider (default, offline)."""

from __future__ import annotations

from propintelli.extraction.llm.base import LlmExtraction


class NoneProvider:
    """A provider that extracts nothing.

    Used when no LLM backend is configured. It lets the engine treat the LLM
    layer uniformly while keeping the pipeline fully deterministic and offline.
    """

    name = "none"

    def extract(self, text: str) -> LlmExtraction:
        """Return an empty extraction regardless of input.

        Parameters
        ----------
        text : str
            Ignored.

        Returns
        -------
        LlmExtraction
            An empty extraction.
        """
        return LlmExtraction()
