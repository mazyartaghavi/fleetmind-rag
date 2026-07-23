from __future__ import annotations

import pytest

from fleetmind_rag.routing import STRATEGIES, RetrievalStrategyRouter


@pytest.fixture
def router() -> RetrievalStrategyRouter:
    return RetrievalStrategyRouter()


def test_analyze_rejects_empty_query(router: RetrievalStrategyRouter) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        router.analyze("   ")


def test_analyze_rejects_nonlexical_query(router: RetrievalStrategyRouter) -> None:
    with pytest.raises(ValueError, match="lexical term"):
        router.analyze("!!!")


def test_analyze_normalizes_whitespace(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("  battery   warning  ")

    assert signals.normalized_query == "battery warning"


def test_analyze_tokenizes_case_insensitively(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("Battery WARNING")

    assert signals.tokens == ("battery", "warning")


def test_analyze_preserves_hyphenated_tokens(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("engine-temperature warning")

    assert signals.tokens[0] == "engine-temperature"


def test_analyze_extracts_quoted_phrases(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("find \"sidewall bulge\" and 'deep cut'")

    assert signals.quoted_phrases == ("sidewall bulge", "deep cut")


def test_analyze_deduplicates_quoted_phrases(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze('"sidewall bulge" or "sidewall bulge"')

    assert signals.quoted_phrases == ("sidewall bulge",)


def test_analyze_extracts_code_like_identifier(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("error code P0420")

    assert signals.exact_identifiers == ("P0420",)


def test_analyze_does_not_treat_plain_year_as_identifier(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("2026 maintenance schedule")

    assert signals.exact_identifiers == ()


def test_analyze_detects_conceptual_cues(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze("What does overheating mean?")

    assert "what does" in signals.conceptual_cues


def test_analyze_detects_conditional_and_action_cues(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("If smoke appears, must I stop safely?")

    assert "if" in signals.conditional_cues
    assert "must i" in signals.action_cues
    assert "stop" in signals.action_cues


def test_analyze_detects_safety_and_domain_cues(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("battery warning with smoke")

    assert "warning" in signals.safety_cues
    assert "smoke" in signals.safety_cues
    assert "battery" in signals.domain_cues


def test_single_word_cues_require_word_boundaries(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("moving through the entire manual")

    assert "vin" not in signals.exact_lookup_cues
    assert "tire" not in signals.domain_cues


def test_conditional_cue_does_not_match_inside_another_word(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("gift inventory")

    assert "if" not in signals.conditional_cues


def test_word_boundary_matching_remains_case_insensitive(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("VIN lookup for a VEHICLE")

    assert "vin" in signals.exact_lookup_cues
    assert "vehicle" in signals.domain_cues


def test_analyze_counts_multiple_clauses(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze(
        "stop the vehicle; switch off the engine; contact maintenance"
    )

    assert signals.clause_count >= 3


def test_complexity_detects_long_query(router: RetrievalStrategyRouter) -> None:
    signals = router.analyze(
        "driver asks whether the vehicle may continue after smoke and dimming "
        "lights appear"
    )

    assert signals.is_complex


def test_complexity_detects_conditional_action_pair(
    router: RetrievalStrategyRouter,
) -> None:
    signals = router.analyze("if smoke appears should i stop")

    assert signals.is_complex


def test_error_code_routes_to_sparse(router: RetrievalStrategyRouter) -> None:
    decision = router.route("error code P0420")

    assert decision.strategy == "sparse"
    assert decision.confidence == "high"


def test_vin_lookup_routes_to_sparse(router: RetrievalStrategyRouter) -> None:
    decision = router.route("find VIN 1HGCM82633A004352")

    assert decision.strategy == "sparse"


def test_quoted_phrase_routes_to_sparse(router: RetrievalStrategyRouter) -> None:
    decision = router.route('find "sidewall bulge"')

    assert decision.strategy == "sparse"


def test_part_number_routes_to_sparse(router: RetrievalStrategyRouter) -> None:
    decision = router.route("locate part number AB-1207")

    assert decision.strategy == "sparse"


def test_conceptual_meaning_query_routes_to_dense(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("What does overheating mean?")

    assert decision.strategy == "dense"


def test_explanation_query_routes_to_dense(router: RetrievalStrategyRouter) -> None:
    decision = router.route("Explain why a charging system can fail")

    assert decision.strategy == "dense"


def test_conceptual_overheating_query_does_not_become_safety_reranked(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("What does an overheating condition mean?")

    assert decision.strategy == "dense"


def test_mixed_operational_terms_route_to_hybrid(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("battery warning smoke smell")

    assert decision.strategy == "hybrid"


def test_short_domain_query_routes_to_hybrid(router: RetrievalStrategyRouter) -> None:
    decision = router.route("vehicle maintenance")

    assert decision.strategy == "hybrid"


def test_unclassified_query_uses_hybrid_fallback(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("available information")

    assert decision.strategy == "hybrid"
    assert "safe default" in decision.reason


def test_conditional_safety_question_routes_to_reranked(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route(
        "If the battery warning is accompanied by smoke, may the driver continue "
        "the trip or must they stop safely?"
    )

    assert decision.strategy == "reranked"
    assert decision.confidence == "high"


def test_operational_action_question_routes_to_reranked(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("What should I do when a red engine warning appears?")

    assert decision.strategy == "reranked"


def test_multi_clause_safety_workflow_routes_to_reranked(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route(
        "When smoke appears, stop the vehicle; switch off the engine; and contact "
        "fleet maintenance"
    )

    assert decision.strategy == "reranked"


def test_exact_identifier_outweighs_conceptual_wording(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("What does error code P0420 mean?")

    assert decision.strategy == "sparse"


def test_decision_contains_all_strategy_scores(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("battery warning smoke smell")

    assert tuple(score.strategy for score in decision.scores) == STRATEGIES


def test_selected_score_matches_strategy_score(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("battery warning smoke smell")
    selected = next(
        score for score in decision.scores if score.strategy == decision.strategy
    )

    assert decision.selected_score == selected.score


def test_selected_reason_is_explainable(router: RetrievalStrategyRouter) -> None:
    decision = router.route("error code P0420")

    assert "identifier" in decision.reason


def test_route_returns_normalized_query(router: RetrievalStrategyRouter) -> None:
    decision = router.route("  battery   warning  ")

    assert decision.query == "battery warning"


def test_route_is_case_insensitive(router: RetrievalStrategyRouter) -> None:
    lower = router.route("battery warning smoke smell")
    upper = router.route("BATTERY WARNING SMOKE SMELL")

    assert lower.strategy == upper.strategy
    assert lower.selected_score == upper.selected_score


def test_route_is_deterministic(router: RetrievalStrategyRouter) -> None:
    first = router.route("battery warning smoke smell")
    second = router.route("battery warning smoke smell")

    assert first == second


def test_strategy_scores_are_non_negative(router: RetrievalStrategyRouter) -> None:
    decision = router.route("complex fleet warning query")

    assert all(score.score >= 0 for score in decision.scores)


def test_medium_confidence_is_reported_for_moderate_margin(
    router: RetrievalStrategyRouter,
) -> None:
    decision = router.route("vehicle maintenance")

    assert decision.confidence in {"medium", "high"}


def test_route_rejects_empty_query(router: RetrievalStrategyRouter) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        router.route("\t\n")
