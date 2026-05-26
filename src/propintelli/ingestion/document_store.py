"""Bronze layer — immutable raw-document store.

Each ingested document is assigned a stable UUID, written verbatim under the
Bronze directory together with a manifest (original filename, SHA-256 content
hash, size, ingestion timestamp). The content hash supports deduplication and
provenance. In production this maps directly onto Azure Blob Storage with an
Event Grid trigger; the local filesystem implementation keeps the same contract.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from propintelli.errors import IngestionError
from propintelli.logging_setup import get_logger

logger = get_logger(__name__)

_MANIFEST_NAME = "manifest.json"
# Manifest source-filename keys across producers (Python snake_case, C# camelCase).
_SOURCE_KEYS = ("source_document", "sourceDocument", "SourceDocument")


class BronzeEntry(BaseModel):
    """A document already resident in the Bronze store.

    Attributes
    ----------
    document_id : str
        The store-assigned identifier (the document's directory name).
    stored_path : Path
        Path to the stored raw file.
    source_document : str
        Original filename, recovered from the manifest when available.
    """

    document_id: str
    stored_path: Path
    source_document: str


class BronzeDocument(BaseModel):
    """Metadata describing a document stored in the Bronze layer.

    Attributes
    ----------
    document_id : str
        Stable identifier assigned at ingestion.
    source_document : str
        Original filename.
    stored_path : Path
        Absolute path to the stored raw file.
    sha256 : str
        Hex SHA-256 digest of the file contents.
    size_bytes : int
        File size in bytes.
    received_at : datetime
        UTC ingestion timestamp.
    """

    document_id: str
    source_document: str
    stored_path: Path
    sha256: str
    size_bytes: int
    received_at: datetime


class DocumentStore:
    """A filesystem-backed Bronze store for raw documents."""

    def __init__(self, bronze_dir: Path) -> None:
        """Create the store rooted at ``bronze_dir``.

        Parameters
        ----------
        bronze_dir : Path
            Directory under which raw documents and manifests are written.
        """
        self._root = bronze_dir
        self._root.mkdir(parents=True, exist_ok=True)

    def ingest_bytes(self, data: bytes, filename: str) -> BronzeDocument:
        """Store raw bytes as a new Bronze document.

        Parameters
        ----------
        data : bytes
            The raw file contents.
        filename : str
            Original filename (used for the stored file's suffix and manifest).

        Returns
        -------
        BronzeDocument
            Metadata for the stored document.

        Raises
        ------
        IngestionError
            If the payload is empty or cannot be written.
        """
        if not data:
            raise IngestionError(f"Refusing to ingest empty document {filename!r}")

        document_id = uuid.uuid4().hex
        suffix = Path(filename).suffix or ".bin"
        target_dir = self._root / document_id
        stored_path = target_dir / f"original{suffix}"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            stored_path.write_bytes(data)
            document = BronzeDocument(
                document_id=document_id,
                source_document=Path(filename).name,
                stored_path=stored_path,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                received_at=datetime.now(UTC),
            )
            (target_dir / _MANIFEST_NAME).write_text(
                document.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise IngestionError(
                f"Failed to store document {filename!r}: {exc}", document_id=document_id
            ) from exc

        logger.info(
            "document_ingested",
            extra={"document_id": document_id, "source_document": document.source_document},
        )
        return document

    def iter_documents(self) -> Iterator[BronzeEntry]:
        """Yield every document resident in the Bronze store.

        Enumerates documents written by *any* producer — the Python pipeline or
        the C# ingestion API — by scanning for the stored ``original.*`` file in
        each document directory. The original filename is read from the manifest
        when present, tolerating both snake_case and camelCase manifest keys.

        Yields
        ------
        BronzeEntry
            One entry per stored document, in directory-name order.
        """
        if not self._root.exists():
            return
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir():
                continue
            originals = sorted(directory.glob("original.*"))
            if not originals:
                continue
            stored_path = originals[0]
            yield BronzeEntry(
                document_id=directory.name,
                stored_path=stored_path,
                source_document=self._read_source_document(directory, stored_path.name),
            )

    @staticmethod
    def _read_source_document(directory: Path, fallback: str) -> str:
        """Recover the original filename from a manifest, or fall back."""
        manifest = directory / _MANIFEST_NAME
        if not manifest.exists():
            return fallback
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return fallback
        for key in _SOURCE_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        return fallback

    def ingest_path(self, path: Path) -> BronzeDocument:
        """Store a document read from a local path.

        Parameters
        ----------
        path : Path
            Path to the source file.

        Returns
        -------
        BronzeDocument
            Metadata for the stored document.

        Raises
        ------
        IngestionError
            If the file does not exist or cannot be read.
        """
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise IngestionError(f"Cannot read document at {path}: {exc}") from exc
        return self.ingest_bytes(data, path.name)
