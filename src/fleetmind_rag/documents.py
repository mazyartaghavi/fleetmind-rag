from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

_HEADING_PATTERN = re.compile(r"^(?P<marker>#{1,6})\s+(?P<title>.+?)\s*$")
_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A normalized UTF-8 text document loaded from disk."""

    document_id: str
    source_name: str
    text: str


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A logical section extracted from a source document."""

    section_id: str
    document_id: str
    ordinal: int
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A deterministic word-window chunk produced from one section."""

    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    ordinal: int
    text: str
    word_count: int
    start_word: int
    end_word: int


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """The complete result of loading, sectioning, and chunking a document."""

    document: SourceDocument
    sections: tuple[DocumentSection, ...]
    chunks: tuple[DocumentChunk, ...]


def normalize_document_text(text: str) -> str:
    """Normalize line endings, horizontal whitespace, and blank lines."""

    if "\x00" in text:
        raise ValueError("Document text must not contain null bytes.")

    normalized_lines: list[str] = []
    previous_line_was_blank = False

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        clean_line = _HORIZONTAL_WHITESPACE_PATTERN.sub(" ", raw_line).strip()

        if not clean_line:
            if normalized_lines and not previous_line_was_blank:
                normalized_lines.append("")
            previous_line_was_blank = True
            continue

        normalized_lines.append(clean_line)
        previous_line_was_blank = False

    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()

    normalized_text = "\n".join(normalized_lines)

    if not normalized_text:
        raise ValueError("Document text must not be empty.")

    return normalized_text


def load_text_document(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> SourceDocument:
    """Load and normalize one text document from disk."""

    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"Document file does not exist: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Document path is not a file: {source_path}")

    normalized_text = normalize_document_text(source_path.read_text(encoding=encoding))
    digest = sha256(normalized_text.encode("utf-8")).hexdigest()[:16]

    return SourceDocument(
        document_id=f"doc-{digest}",
        source_name=source_path.name,
        text=normalized_text,
    )


def split_document_sections(
    document: SourceDocument,
    *,
    default_title: str | None = None,
) -> tuple[DocumentSection, ...]:
    """Split a document at Markdown-style headings."""

    fallback_title = _clean_title(
        default_title if default_title is not None else Path(document.source_name).stem
    )
    sections: list[DocumentSection] = []
    current_title = fallback_title
    current_lines: list[str] = []

    def append_current_section() -> None:
        section_text = _normalize_optional_text("\n".join(current_lines))
        if section_text is None:
            return

        ordinal = len(sections) + 1
        sections.append(
            DocumentSection(
                section_id=f"{document.document_id}-section-{ordinal:03d}",
                document_id=document.document_id,
                ordinal=ordinal,
                title=current_title,
                text=section_text,
            )
        )

    for line in document.text.splitlines():
        heading_match = _HEADING_PATTERN.fullmatch(line)

        if heading_match is None:
            current_lines.append(line)
            continue

        append_current_section()
        current_lines = []
        current_title = _clean_title(heading_match.group("title"))

    append_current_section()

    if sections:
        return tuple(sections)

    return (
        DocumentSection(
            section_id=f"{document.document_id}-section-001",
            document_id=document.document_id,
            ordinal=1,
            title=fallback_title,
            text=document.text,
        ),
    )


def chunk_document_sections(
    sections: Sequence[DocumentSection],
    *,
    chunk_size_words: int = 180,
    overlap_words: int = 30,
) -> tuple[DocumentChunk, ...]:
    """Create deterministic overlapping word-window chunks."""

    if chunk_size_words <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap_words < 0:
        raise ValueError("Chunk overlap must not be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks: list[DocumentChunk] = []
    step_size = chunk_size_words - overlap_words

    for section in sections:
        words = section.text.split()

        if not words:
            continue

        start_word = 0
        section_chunk_ordinal = 1

        while start_word < len(words):
            end_word = min(start_word + chunk_size_words, len(words))
            chunk_words = words[start_word:end_word]

            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{section.section_id}-chunk-{section_chunk_ordinal:03d}"
                    ),
                    document_id=section.document_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    ordinal=section_chunk_ordinal,
                    text=" ".join(chunk_words),
                    word_count=len(chunk_words),
                    start_word=start_word,
                    end_word=end_word,
                )
            )

            if end_word == len(words):
                break

            start_word += step_size
            section_chunk_ordinal += 1

    if not chunks:
        raise ValueError("At least one non-empty section is required.")

    return tuple(chunks)


def ingest_text_document(
    path: str | Path,
    *,
    default_title: str | None = None,
    chunk_size_words: int = 180,
    overlap_words: int = 30,
    encoding: str = "utf-8",
) -> IngestedDocument:
    """Load, section, and chunk one text document."""

    document = load_text_document(path, encoding=encoding)
    sections = split_document_sections(document, default_title=default_title)
    chunks = chunk_document_sections(
        sections,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    return IngestedDocument(
        document=document,
        sections=sections,
        chunks=chunks,
    )


def _clean_title(title: str) -> str:
    clean_title = _HORIZONTAL_WHITESPACE_PATTERN.sub(" ", title).strip()

    if not clean_title:
        raise ValueError("Section title must not be empty.")

    return clean_title


def _normalize_optional_text(text: str) -> str | None:
    if not text.strip():
        return None

    return normalize_document_text(text)
