from typing import Any, Dict

from fastmcp import FastMCP

from services.abnormal_decision import perform_abnormal_decision

__all__ = ["register_all_tools"]


def abnormal_decision(
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
) -> Dict[str, Any]:
    """
    사용자의 이상 판정 요청을 처리하는 통합 도구입니다.
    시계열 분석, 유사 사례 검색, 최종 응답 생성을 일괄 수행합니다.
    """
    return perform_abnormal_decision(
        station_id=station_id,
        element=element,
        start_time=start_time,
        end_time=end_time,
    )


def register_all_tools(mcp: FastMCP) -> None:
    mcp.tool(
        description=(
            "특정 측정소의 시계열 데이터를 분석하고, 과거 유사 사례와 비교하여 "
            "이상 여부를 판정합니다. "
            "필수 인자: station_id, element, start_time, end_time"
        )
    )(abnormal_decision)
