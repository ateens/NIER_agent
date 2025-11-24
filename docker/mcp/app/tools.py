from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from config import get_settings
from services import (
    build_insight_payload,
    orchestrate_response,
    perform_timeseries_analysis,
)

__all__ = ["register_all_tools"]


def abnormal_decision(
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    window_size: int = 6,
    comparison_type: str = "dtw",
    missing_value_policy: str = "zero",
    collection: Optional[str] = None,
    device: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    사용자의 이상 판정 요청을 처리하는 통합 도구입니다.
    시계열 분석, 유사 사례 검색, 최종 응답 생성을 일괄 수행합니다.
    """
    settings = get_settings()

    # 1. Time Series Analysis
    analysis_result = perform_timeseries_analysis(
        settings,
        station_id=station_id,
        element=element,
        start_time=start_time,
        end_time=end_time,
        window_size=window_size,
        comparison_type=comparison_type,
        missing_value_policy=missing_value_policy,
        # Default values for others
        double_sequence=None,
        additional_days=None,
        include_related=True,
        compute_similarity=True,
        max_related=None,
    )

    # 2. Insight Retrieval
    # Extract original series with values for embedding
    original_series = analysis_result.get("original", {})
    # Ensure values are present (we modified timeseries_service to keep them)
    
    insight_result = build_insight_payload(
        values=[original_series],  # Pass the dict containing 'values'
        element=element,
        collection=collection,
        perform_embedding=True,
        perform_search=True,
        device=device,
        filters=filters,
        embedding=None,
    )

    # 3. Response Orchestration
    orchestration_result = orchestrate_response(
        response_type="analysis",
        query={
            "station_id": station_id,
            "element": element,
            "start_time": start_time,
            "end_time": end_time,
        },
        raw_data=analysis_result,
        insight=insight_result,
        history=history,
        messages=messages,
    )

    # Filter output
    return {
        "answer": orchestration_result.get("answer"),
        "graph_image": orchestration_result.get("graph_image"),
    }


def register_all_tools(mcp: FastMCP) -> None:
    mcp.tool(
        description=(
            "특정 측정소의 시계열 데이터를 분석하고, 과거 유사 사례와 비교하여 이상 여부를 판정합니다."
            "필수 인자: station_id, element, start_time, end_time"
        )
    )(abnormal_decision)
