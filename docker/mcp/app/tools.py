from typing import Any, Dict, List, Literal, Optional, Sequence

from fastmcp import FastMCP

from config import get_settings
from services import (
    build_insight_payload,
    orchestrate_response,
    perform_timeseries_analysis,
)

__all__ = ["register_all_tools"]


def timeseries_analysis(
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
    *,
    double_sequence: Optional[bool] = None,
    additional_days: Optional[int] = None,
    include_related: bool = True,
    include_context: bool = True,
    compute_similarity: bool = True,
    comparison_type: str = "dtw",
    max_related: Optional[int] = None,
    window_size: int = 6,
    missing_value_policy: str = "zero",
) -> Dict[str, Any]:
    settings = get_settings()
    return perform_timeseries_analysis(
        settings,
        station_id=station_id,
        element=element,
        start_time=start_time,
        end_time=end_time,
        double_sequence=double_sequence,
        additional_days=additional_days,
        include_related=include_related,
        include_context=include_context,
        compute_similarity=compute_similarity,
        comparison_type=comparison_type,
        max_related=max_related,
        window_size=window_size,
        missing_value_policy=missing_value_policy,
    )


def insight_retrieval(
    values: Optional[Any] = None,
    *,
    element: Optional[str] = None,
    collection: Optional[str] = None,
    perform_embedding: bool = True,
    perform_search: bool = True,
    embedding: Optional[List[float]] = None,
    device: Optional[str] = None,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_insight_payload(
        values,
        element=element,
        collection=collection,
        perform_embedding=perform_embedding,
        perform_search=perform_search,
        embedding=embedding,
        device=device,
        top_k=top_k,
        filters=filters,
    )


def response_orchestration(
    *,
    response_type: Literal["analysis", "general", "log_only"] = "analysis",
    query: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    comparisons: Optional[List[Dict[str, Any]]] = None,
    neighbors: Optional[List[Dict[str, Any]]] = None,
    station_context: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    log_stage: Optional[str] = None,
    log_payload: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    return orchestrate_response(
        response_type=response_type,
        query=query,
        raw_data=raw_data,
        comparisons=comparisons,
        neighbors=neighbors,
        station_context=station_context,
        history=history,
        messages=messages,
        log_stage=log_stage,
        log_payload=log_payload,
        latency_ms=latency_ms,
    )


def register_all_tools(mcp: FastMCP) -> None:
    mcp.tool(
        description=(
            "원본 및 연관 측정소 시계열을 조회하고 필요 시 맥락 정보와 유사도 지표를 함께 제공합니다. "
            "station_id, element, start_time, end_time은 필수이며, include_related / include_context / "
            "compute_similarity 옵션으로 단계를 개별적으로 제어할 수 있습니다."
        )
    )(timeseries_analysis)

    mcp.tool(
        description=(
            "T-Rep 임베딩 생성과 벡터 검색을 하나의 호출로 처리합니다. "
            "values 또는 embedding을 입력으로 받아 perform_embedding / perform_search 옵션을 조정하세요."
        )
    )(insight_retrieval)

    mcp.tool(
        description=(
            "타임시리즈 분석 응답, 일반 질의, 로깅을 단일 엔드포인트로 처리합니다. "
            "response_type을 analysis/general/log_only 중 선택하고 필요한 페이로드를 전달하세요."
        )
    )(response_orchestration)
