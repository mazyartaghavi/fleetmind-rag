from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fleetmind_rag.adaptive_retrieval import (
    AdaptiveRetrievalConfig,
    AdaptiveRetrievalOutcome,
    DeterministicQueryRewriter,
    QueryRewrite,
    RoutedRetrievalExecutorLike,
)
from fleetmind_rag.agent_state import RetrievalAgentState
from fleetmind_rag.retrieval_quality import (
    RetrievalQualityAssessment,
    RetrievalQualityChecker,
)
from fleetmind_rag.routed_retrieval import (
    RoutedRetrievalRequest,
    RoutedRetrievalResult,
)

LangGraphRoute = Literal["complete", "rewrite", "fail"]


class AdaptiveRetrievalGraphState(TypedDict):
    """Shared state passed between LangGraph retrieval nodes."""

    agent_state: RetrievalAgentState
    config: AdaptiveRetrievalConfig
    assessments: tuple[RetrievalQualityAssessment, ...]
    rewrites: tuple[QueryRewrite, ...]
    result: NotRequired[RoutedRetrievalResult]
    assessment: NotRequired[RetrievalQualityAssessment]


class LangGraphAdaptiveRetrievalWorkflow:
    """Explicit LangGraph orchestration for adaptive retrieval."""

    def __init__(
        self,
        executor: RoutedRetrievalExecutorLike,
        *,
        quality_checker: RetrievalQualityChecker | None = None,
        query_rewriter: DeterministicQueryRewriter | None = None,
    ) -> None:
        """Initialize graph dependencies and compile the workflow."""

        self._executor = executor
        self._quality_checker = quality_checker or RetrievalQualityChecker()
        self._query_rewriter = query_rewriter or DeterministicQueryRewriter()
        self._graph = self._compile_graph()

    @property
    def graph(
        self,
    ) -> CompiledStateGraph[
        AdaptiveRetrievalGraphState,
        None,
        AdaptiveRetrievalGraphState,
        AdaptiveRetrievalGraphState,
    ]:
        """Return the compiled and directly invokable LangGraph."""

        return self._graph

    def draw_mermaid(self) -> str:
        """Return an offline Mermaid description of the graph topology."""

        return self._graph.get_graph().draw_mermaid()

    def run(
        self,
        query: str,
        *,
        config: AdaptiveRetrievalConfig | None = None,
    ) -> AdaptiveRetrievalOutcome:
        """Invoke the graph and return the domain-level final outcome."""

        resolved_config = config or AdaptiveRetrievalConfig()
        initial_state: AdaptiveRetrievalGraphState = {
            "agent_state": RetrievalAgentState.start(
                query,
                max_attempts=resolved_config.max_attempts,
            ),
            "config": resolved_config,
            "assessments": (),
            "rewrites": (),
        }
        invoke_config: RunnableConfig = {
            "recursion_limit": max(
                25,
                resolved_config.max_attempts * 4 + 5,
            )
        }
        final_state = self._graph.invoke(
            initial_state,
            config=invoke_config,
        )

        return AdaptiveRetrievalOutcome(
            state=final_state["agent_state"],
            assessments=final_state["assessments"],
            rewrites=final_state["rewrites"],
        )

    def _compile_graph(
        self,
    ) -> CompiledStateGraph[
        AdaptiveRetrievalGraphState,
        None,
        AdaptiveRetrievalGraphState,
        AdaptiveRetrievalGraphState,
    ]:
        builder = StateGraph(AdaptiveRetrievalGraphState)
        builder.add_node(
            "retrieve",
            RunnableLambda[AdaptiveRetrievalGraphState, AdaptiveRetrievalGraphState](
                self._retrieve_node
            ),
        )
        builder.add_node(
            "assess",
            RunnableLambda[AdaptiveRetrievalGraphState, AdaptiveRetrievalGraphState](
                self._assess_node
            ),
        )
        builder.add_node(
            "rewrite",
            RunnableLambda[AdaptiveRetrievalGraphState, AdaptiveRetrievalGraphState](
                self._rewrite_node
            ),
        )
        builder.add_node(
            "complete",
            RunnableLambda[AdaptiveRetrievalGraphState, AdaptiveRetrievalGraphState](
                self._complete_node
            ),
        )
        builder.add_node(
            "fail",
            RunnableLambda[AdaptiveRetrievalGraphState, AdaptiveRetrievalGraphState](
                self._fail_node
            ),
        )

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {
                "complete": "complete",
                "rewrite": "rewrite",
                "fail": "fail",
            },
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("complete", END)
        builder.add_edge("fail", END)
        return builder.compile(name="fleetmind-adaptive-retrieval")

    def _retrieve_node(
        self,
        graph_state: AdaptiveRetrievalGraphState,
    ) -> AdaptiveRetrievalGraphState:
        agent_state = graph_state["agent_state"].begin_retrieval()
        config = graph_state["config"]
        result = self._executor.execute(
            RoutedRetrievalRequest(
                query=agent_state.current_query,
                limit=config.limit,
                candidate_limit=config.candidate_limit,
                score_threshold=config.score_threshold,
                metadata_filter=config.metadata_filter,
            )
        )
        agent_state = agent_state.record_retrieval(result)
        updated_state = graph_state.copy()
        updated_state["agent_state"] = agent_state
        updated_state["result"] = result
        return updated_state

    def _assess_node(
        self,
        graph_state: AdaptiveRetrievalGraphState,
    ) -> AdaptiveRetrievalGraphState:
        result = _require_result(graph_state)
        assessment = self._quality_checker.assess(result)
        updated_state = graph_state.copy()
        updated_state["assessment"] = assessment
        updated_state["assessments"] = (
            *graph_state["assessments"],
            assessment,
        )
        return updated_state

    @staticmethod
    def _route_after_assessment(
        graph_state: AdaptiveRetrievalGraphState,
    ) -> LangGraphRoute:
        assessment = _require_assessment(graph_state)

        if assessment.should_accept:
            return "complete"

        if graph_state["agent_state"].can_retry:
            return "rewrite"

        return "fail"

    def _rewrite_node(
        self,
        graph_state: AdaptiveRetrievalGraphState,
    ) -> AdaptiveRetrievalGraphState:
        result = _require_result(graph_state)
        assessment = _require_assessment(graph_state)
        agent_state = graph_state["agent_state"].request_rewrite(
            "; ".join(assessment.reasons)
        )
        rewrite = self._query_rewriter.rewrite(
            result,
            assessment,
            after_attempt=agent_state.attempt_count,
        )
        agent_state = agent_state.apply_rewrite(rewrite.rewritten_query)
        updated_state = graph_state.copy()
        updated_state["agent_state"] = agent_state
        updated_state["rewrites"] = (
            *graph_state["rewrites"],
            rewrite,
        )
        return updated_state

    @staticmethod
    def _complete_node(
        graph_state: AdaptiveRetrievalGraphState,
    ) -> AdaptiveRetrievalGraphState:
        assessment = _require_assessment(graph_state)
        agent_state = graph_state["agent_state"].complete(
            "Retrieval quality checks passed with score "
            f"{assessment.quality_score:.4f}."
        )
        updated_state = graph_state.copy()
        updated_state["agent_state"] = agent_state
        return updated_state

    @staticmethod
    def _fail_node(
        graph_state: AdaptiveRetrievalGraphState,
    ) -> AdaptiveRetrievalGraphState:
        assessment = _require_assessment(graph_state)
        agent_state = graph_state["agent_state"]
        failed_state = agent_state.fail(
            "Retrieval quality remained insufficient after "
            f"{agent_state.attempt_count} attempts: "
            f"{'; '.join(assessment.reasons)}"
        )
        updated_state = graph_state.copy()
        updated_state["agent_state"] = failed_state
        return updated_state


def _require_result(
    graph_state: AdaptiveRetrievalGraphState,
) -> RoutedRetrievalResult:
    result = graph_state.get("result")

    if result is None:
        raise RuntimeError("LangGraph state does not contain a retrieval result")

    return result


def _require_assessment(
    graph_state: AdaptiveRetrievalGraphState,
) -> RetrievalQualityAssessment:
    assessment = graph_state.get("assessment")

    if assessment is None:
        raise RuntimeError("LangGraph state does not contain a quality assessment")

    return assessment
