from pathlib import Path

import pytest

from fleetmind_rag.documents import (
    DocumentSection,
    SourceDocument,
    chunk_document_sections,
    ingest_text_document,
    load_text_document,
    normalize_document_text,
    split_document_sections,
)


def test_normalize_document_text_standardizes_spacing_and_blank_lines() -> None:
    text = "  Fleet\toperations  \r\n\r\n\r\n  Engine   checks  \r"

    assert normalize_document_text(text) == "Fleet operations\n\nEngine checks"


def test_normalize_document_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_document_text(" \n\t ")


def test_normalize_document_text_rejects_null_bytes() -> None:
    with pytest.raises(ValueError, match="null bytes"):
        normalize_document_text("fleet\x00manual")


def test_load_text_document_creates_stable_content_identifier(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("Fleet manual", encoding="utf-8")
    second_path.write_text("Fleet manual", encoding="utf-8")

    first = load_text_document(first_path)
    second = load_text_document(second_path)

    assert first.document_id == second.document_id
    assert first.source_name == "first.txt"
    assert first.text == "Fleet manual"


def test_load_text_document_rejects_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_text_document(missing_path)


def test_split_document_sections_uses_markdown_headings() -> None:
    document = SourceDocument(
        document_id="doc-test",
        source_name="manual.md",
        text=(
            "# Engine Warnings\nPark safely.\n\n## Tire Pressure\nInspect the tires."
        ),
    )

    sections = split_document_sections(document)

    assert [section.title for section in sections] == [
        "Engine Warnings",
        "Tire Pressure",
    ]
    assert [section.text for section in sections] == [
        "Park safely.",
        "Inspect the tires.",
    ]
    assert [section.section_id for section in sections] == [
        "doc-test-section-001",
        "doc-test-section-002",
    ]


def test_split_document_sections_preserves_preamble() -> None:
    document = SourceDocument(
        document_id="doc-test",
        source_name="manual.md",
        text=("General fleet guidance.\n\n# Engine Warnings\nPark safely."),
    )

    sections = split_document_sections(document, default_title="Fleet Overview")

    assert [section.title for section in sections] == [
        "Fleet Overview",
        "Engine Warnings",
    ]


def test_split_document_sections_falls_back_without_headings() -> None:
    document = SourceDocument(
        document_id="doc-test",
        source_name="fleet_manual.txt",
        text="Inspect every vehicle before departure.",
    )

    sections = split_document_sections(document)

    assert len(sections) == 1
    assert sections[0].title == "fleet_manual"
    assert sections[0].text == document.text


def test_chunk_document_sections_creates_exact_overlap() -> None:
    section = DocumentSection(
        section_id="doc-test-section-001",
        document_id="doc-test",
        ordinal=1,
        title="Operations",
        text="one two three four five six seven",
    )

    chunks = chunk_document_sections(
        [section],
        chunk_size_words=4,
        overlap_words=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [
        (0, 4),
        (2, 6),
        (4, 7),
    ]


def test_chunk_document_sections_uses_deterministic_identifiers() -> None:
    section = DocumentSection(
        section_id="doc-test-section-001",
        document_id="doc-test",
        ordinal=1,
        title="Operations",
        text="one two three four five",
    )

    chunks = chunk_document_sections(
        [section],
        chunk_size_words=3,
        overlap_words=1,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-test-section-001-chunk-001",
        "doc-test-section-001-chunk-002",
    ]
    assert [chunk.ordinal for chunk in chunks] == [1, 2]


@pytest.mark.parametrize(
    ("chunk_size_words", "overlap_words", "message"),
    [
        (0, 0, "greater than zero"),
        (10, -1, "must not be negative"),
        (10, 10, "smaller than chunk size"),
        (10, 11, "smaller than chunk size"),
    ],
)
def test_chunk_document_sections_validates_configuration(
    chunk_size_words: int,
    overlap_words: int,
    message: str,
) -> None:
    section = DocumentSection(
        section_id="doc-test-section-001",
        document_id="doc-test",
        ordinal=1,
        title="Operations",
        text="fleet operations",
    )

    with pytest.raises(ValueError, match=message):
        chunk_document_sections(
            [section],
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )


def test_chunk_document_sections_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="non-empty section"):
        chunk_document_sections([])


def test_ingest_text_document_runs_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "fleet.md"
    path.write_text(
        (
            "# Engine Warnings\n"
            "Park the vehicle safely and record the warning.\n\n"
            "# Tire Pressure\n"
            "Inspect the tire and record the measured pressure."
        ),
        encoding="utf-8",
    )

    result = ingest_text_document(
        path,
        chunk_size_words=5,
        overlap_words=1,
    )

    assert result.document.source_name == "fleet.md"
    assert len(result.sections) == 2
    assert len(result.chunks) == 4
    assert all(
        chunk.document_id == result.document.document_id for chunk in result.chunks
    )
