from typing import Any, Dict, List, Literal, Optional, Sequence


_Z_SCORE_COMMENTS = {
    "평균 유사도": "와의 유사도가 평균 범위 내에 있습니다.",
    "약간 낮은 유사도": "와의 유사도가 약간 낮습니다.",
    "낮은 유사도": "와의 유사도가 평균보다 낮으므로 이상 징후 의심이 됩니다.",
    "매우 낮은 유사도": "와의 유사도가 평균보다 크게 낮아 베이스라인 이상 징후가 강하게 의심됩니다.",
    "데이터 부족": "에 대한 비교 데이터를 확보하지 못했습니다.",
}


def _format_station_section(comparisons: Sequence[Dict[str, Any]]) -> str:
    if not comparisons:
        return "연관 측정소가 없습니다."

    lines: List[str] = []
    for idx, item in enumerate(comparisons, start=1):
        station_id = item.get("station_id", "N/A")
        label = item.get("z_score_range", "데이터 부족")
        comment = _Z_SCORE_COMMENTS.get(label, "에 대한 비교 결과가 부족합니다.")
        lines.append(f"- {idx}순위 연관 측정소 {station_id} {comment}")
    return "\n".join(lines)


def _format_neighbor_section(neighbors: Sequence[Dict[str, Any]]) -> str:
    if not neighbors:
        return "유사 기존 판정 결과가 없습니다."

    lines: List[str] = []
    for item in neighbors:
        station_id = item.get("station_id", "N/A")
        element = item.get("element", "N/A")
        state = item.get("class_label", "분류 불명")
        similarity = item.get("similarity", "정보 부족")
        original_start = item.get("original_start", "N/A")
        original_end = item.get("original_end", "N/A")
        lines.append(
            f"- 기존 판정 결과 {station_id} (성분: {element}, 상태: {state}, 유사도: {similarity}, "
            f"측정일시: {original_start} ~ {original_end})"
        )
    return "\n".join(lines)

__all__ = ["orchestrate_response"]


def orchestrate_response(
    *,
    response_type: Literal["analysis", "general", "log_only"] = "analysis",
    query: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    neighbors: Optional[Sequence[Dict[str, Any]]] = None,
    insight: Optional[Dict[str, Any]] = None,
    station_context: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    log_stage: Optional[str] = None,
    log_payload: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "response_type": response_type,
    }

    log_entry: Optional[Dict[str, Any]] = None
    if log_stage:
        log_entry = {
            "stage": log_stage,
            "payload": log_payload or {},
            "latency_ms": latency_ms,
            "status": "logged",
        }

    answer = ""
    if response_type == "general":
        used_messages = list(messages or history or [])
        metadata["history_length"] = len(used_messages)
        answer = "일반 질의 응답이 여기에 표시됩니다."
    elif response_type == "log_only":
        answer = "로깅이 완료되었습니다."
    else:
        analysis_payload = raw_data or {}
        comparison_entries = list(analysis_payload.get("comparisons") or [])
        insight_payload = insight or {}
        if not insight_payload and isinstance(neighbors, dict):
            insight_payload = neighbors  # backwards compatibility when payload passed via neighbors param

        avg_confidence = analysis_payload.get("avg_confidence")
        if avg_confidence is None and comparison_entries:
            avg_confidence = comparison_entries[0].get("avg_confidence")
        if avg_confidence is None:
            avg_confidence = 1.0

        abnormal_probability = insight_payload.get("abnormal_probability", 0.0)
        neighbor_entries = insight_payload.get("neighbors")
        if neighbor_entries is None and isinstance(neighbors, Sequence):
            neighbor_entries = neighbors
        neighbor_entries = neighbor_entries or []

        station_section = _format_station_section(comparison_entries)
        neighbor_section = _format_neighbor_section(neighbor_entries)

        extreme_difference_comment = ""

        final_abnormal_probability = (
            (abnormal_probability * 0.5)
            + ((1 - float(avg_confidence)) * 50.0)
        )
        final_abnormal_probability = max(0.0, min(100.0, final_abnormal_probability))

        station_section += "\n### 종합 판정 확률 (연관 측정소 및 기존 판정 결과 반영) ###\n"
        station_section += f"\n- 베이스라인 이상 확률: {final_abnormal_probability:.2f}%\n"

        original = analysis_payload.get("original", {})
        element = original.get("element") or (analysis_payload.get("query", {}) or {}).get("element", "N/A")
        station_id = original.get("region") or (analysis_payload.get("query", {}) or {}).get("region", "N/A")
        start_time = original.get("start_time") or (analysis_payload.get("query", {}) or {}).get("start_time", "N/A")
        end_time = original.get("end_time") or (analysis_payload.get("query", {}) or {}).get("end_time", "N/A")

        prompt = (
            "## 데이터 판정 요청 ##\n"
            f"다음은 측정소 {station_id} 에서 {start_time} 부터 {end_time} 까지 수집된 {element} 성분의 비교 결과입니다.\n\n"
            "### 유사한 기존 판정 결과 ###\n"
            f"{neighbor_section}\n\n"
            f"{extreme_difference_comment}"
            f"### 연관 측정소 비교 결과 ###\n{station_section}\n"
            "### 요청 사항 ###\n"
            "위 데이터를 종합적으로 분석하여, 해당 데이터가 정상인지 베이스라인 이상인지 판정하고 이유를 설명해주세요.\n"
            "연관 측정소 비교 결과와 유사한 기존 판정 결과의 지역번호, 성분, 상태, 유사도, 측정일시를 사용자에게 보여주세요\n"
        )

        metadata.update(
            {
                "has_query": query is not None,
                "has_raw_data": bool(analysis_payload),
                "comparison_count": len(comparison_entries),
                "neighbor_count": len(neighbor_entries),
                "avg_confidence": avg_confidence,
                "abnormal_probability": abnormal_probability,
                "final_abnormal_probability": final_abnormal_probability,
            }
        )
        answer = prompt

    return {
        "answer": answer,
        "metadata": metadata,
        "log": log_entry,
    }
