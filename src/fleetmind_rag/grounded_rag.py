from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from fleetmind_rag.ollama import OllamaChatResult
from fleetmind_rag.retrieval import RetrievalResponse
from fleetmind_rag.vector_store import VectorSearchResult

DEFAULT_GROUNDED_SYSTEM_PROMPT = """You are FleetMind, a cautious
fleet-operations assistant.
Answer only from the retrieved evidence supplied in the user prompt.
Use the provided source labels as inline citations, such as [S1].
Never invent maintenance procedures, limits, diagnoses, or citations.
If the evidence is insufficient, reply exactly: INSUFFICIENT_CONTEXT"""

ABSTENTION_ANSWER = (
    "I do not have enough grounded evidence in the indexed fleet documents "
    "to answer this question safely."
)

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class RetrievalClient(Protocol):
    """Structural interface required from a document retrieval service."""

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Return ranked chunks for one query."""


class ChatClient(Protocol):
    """Structural interface required from a chat-generation client."""

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> OllamaChatResult:
        """Generate one complete assistant response."""


@dataclass(frozen=True, slots=True)
class GroundedCitation:
    """One retrieved source exposed with a grounded answer."""

    label: str
    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    """Validated outcome of one grounded-answer request."""

    succeeded: bool
    abstained: bool
    question: str
    answer: str | None
    citations: tuple[GroundedCitation, ...]
    retrieval_model: str | None
    generation_model: str | None
    top_score: float | None
    message: str


class GroundedAnswerService:
    """Generate citation-grounded answers from retrieved fleet documents."""

    def __init__(
        self,
        retrieval_client: RetrievalClient,
        chat_client: ChatClient,
        *,
        minimum_score: float = 0.5,
        max_context_chars: int = 6000,
        system_prompt: str = DEFAULT_GROUNDED_SYSTEM_PROMPT,
    ) -> None:
        if not math.isfinite(minimum_score):
            raise ValueError("The minimum retrieval score must be finite.")

        if max_context_chars < 256:
            raise ValueError(
                "The maximum context size must be at least 256 characters."
            )

        clean_system_prompt = system_prompt.strip()
        if not clean_system_prompt:
            raise ValueError("The grounded-answer system prompt must not be empty.")

        self._retrieval_client = retrieval_client
        self._chat_client = chat_client
        self._minimum_score = minimum_score
        self._max_context_chars = max_context_chars
        self._system_prompt = clean_system_prompt

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
    ) -> GroundedAnswerResult:
        """Retrieve evidence and generate a citation-grounded answer."""

        clean_question = question.strip()

        if not clean_question:
            raise ValueError("The grounded-answer question must not be empty.")

        if limit <= 0:
            raise ValueError("The retrieval limit must be greater than zero.")

        retrieval = self._retrieval_client.search(
            clean_question,
            limit=limit,
            score_threshold=None,
        )
        top_score = self._top_score(retrieval.matches)
        relevant_matches = self._relevant_matches(retrieval.matches)

        if not relevant_matches:
            return self._abstention_result(
                question=clean_question,
                retrieval_model=retrieval.embedding_model,
                generation_model=None,
                top_score=top_score,
                message=(
                    "No retrieved chunk met the minimum grounding score; "
                    "generation was skipped."
                ),
            )

        context, citations = self._build_context(relevant_matches)
        prompt = self._build_user_prompt(clean_question, context)
        chat_result = self._chat_client.chat(
            prompt,
            system_prompt=self._system_prompt,
        )

        if not chat_result.succeeded:
            return GroundedAnswerResult(
                succeeded=False,
                abstained=False,
                question=clean_question,
                answer=None,
                citations=citations,
                retrieval_model=retrieval.embedding_model,
                generation_model=None,
                top_score=top_score,
                message=f"Grounded generation failed: {chat_result.message}",
            )

        answer = chat_result.content.strip() if chat_result.content is not None else ""
        generation_model = (
            chat_result.model.strip() if chat_result.model is not None else ""
        )

        if not answer:
            return GroundedAnswerResult(
                succeeded=False,
                abstained=False,
                question=clean_question,
                answer=None,
                citations=citations,
                retrieval_model=retrieval.embedding_model,
                generation_model=generation_model or None,
                top_score=top_score,
                message="The successful chat response contained no answer text.",
            )

        if not generation_model:
            return GroundedAnswerResult(
                succeeded=False,
                abstained=False,
                question=clean_question,
                answer=None,
                citations=citations,
                retrieval_model=retrieval.embedding_model,
                generation_model=None,
                top_score=top_score,
                message="The successful chat response contained no model name.",
            )

        if answer.upper() == _INSUFFICIENT_CONTEXT:
            return self._abstention_result(
                question=clean_question,
                retrieval_model=retrieval.embedding_model,
                generation_model=generation_model,
                top_score=top_score,
                message="The generation model reported insufficient context.",
            )

        cited_labels = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        citation_by_label = {citation.label: citation for citation in citations}

        if not cited_labels or any(
            label not in citation_by_label for label in cited_labels
        ):
            fallback_citation = citations[0]
            return GroundedAnswerResult(
                succeeded=True,
                abstained=False,
                question=clean_question,
                answer=f"{fallback_citation.text} [{fallback_citation.label}]",
                citations=(fallback_citation,),
                retrieval_model=retrieval.embedding_model,
                generation_model=generation_model,
                top_score=top_score,
                message=(
                    "The generated answer did not contain valid source citations; "
                    "an extractive fallback was returned."
                ),
            )

        used_citations = tuple(citation_by_label[label] for label in cited_labels)

        return GroundedAnswerResult(
            succeeded=True,
            abstained=False,
            question=clean_question,
            answer=answer,
            citations=used_citations,
            retrieval_model=retrieval.embedding_model,
            generation_model=generation_model,
            top_score=top_score,
            message="Generated a citation-grounded answer.",
        )

    def _relevant_matches(
        self,
        matches: tuple[VectorSearchResult, ...],
    ) -> tuple[VectorSearchResult, ...]:
        relevant: list[VectorSearchResult] = []

        for match in matches:
            if not math.isfinite(match.score):
                raise RuntimeError("A retrieval result contains a non-finite score.")

            if match.score >= self._minimum_score:
                relevant.append(match)

        return tuple(relevant)

    @staticmethod
    def _top_score(matches: tuple[VectorSearchResult, ...]) -> float | None:
        if not matches:
            return None

        scores = tuple(match.score for match in matches)
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("A retrieval result contains a non-finite score.")

        return max(scores)

    def _build_context(
        self,
        matches: tuple[VectorSearchResult, ...],
    ) -> tuple[str, tuple[GroundedCitation, ...]]:
        blocks: list[str] = []
        citations: list[GroundedCitation] = []
        remaining_chars = self._max_context_chars

        for index, match in enumerate(matches, start=1):
            label = f"S{index}"
            header = (
                f"[{label}]\n"
                f"Document ID: {match.document_id}\n"
                f"Section: {match.section_title}\n"
                f"Chunk ID: {match.chunk_id}\n"
                f"Similarity score: {match.score:.4f}\n"
                "Content: "
            )
            separator_size = 2 if blocks else 0
            available_text_chars = remaining_chars - len(header) - separator_size

            if available_text_chars <= 0:
                break

            source_text = match.text.strip()
            if len(source_text) > available_text_chars:
                if blocks:
                    break
                source_text = self._truncate_text(source_text, available_text_chars)

            block = f"{header}{source_text}"
            blocks.append(block)
            citations.append(
                GroundedCitation(
                    label=label,
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    section_id=match.section_id,
                    section_title=match.section_title,
                    text=source_text,
                    score=match.score,
                )
            )
            remaining_chars -= len(block) + separator_size

        if not blocks:
            raise RuntimeError("The context budget could not fit any retrieved source.")

        return "\n\n".join(blocks), tuple(citations)

    @staticmethod
    def _truncate_text(text: str, maximum_chars: int) -> str:
        if maximum_chars <= 3:
            return text[:maximum_chars]

        return f"{text[: maximum_chars - 3].rstrip()}..."

    @staticmethod
    def _build_user_prompt(question: str, context: str) -> str:
        return (
            "Answer the fleet-operations question using only the retrieved evidence.\n"
            "Cite every supported claim with the exact source labels shown below.\n"
            "If the evidence is insufficient, reply exactly: "
            f"{_INSUFFICIENT_CONTEXT}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{context}"
        )

    @staticmethod
    def _abstention_result(
        *,
        question: str,
        retrieval_model: str,
        generation_model: str | None,
        top_score: float | None,
        message: str,
    ) -> GroundedAnswerResult:
        return GroundedAnswerResult(
            succeeded=True,
            abstained=True,
            question=question,
            answer=ABSTENTION_ANSWER,
            citations=(),
            retrieval_model=retrieval_model,
            generation_model=generation_model,
            top_score=top_score,
            message=message,
        )
