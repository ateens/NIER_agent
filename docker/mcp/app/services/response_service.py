from typing import Any, Dict, Literal, Optional, Sequence

__all__ = ["orchestrate_response"]


def orchestrate_response(
    *,
    response_type: Literal["analysis", "general", "log_only"] = "analysis",
    query: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    comparisons: Optional[Sequence[Dict[str, Any]]] = None,
    neighbors: Optional[Sequence[Dict[str, Any]]] = None,
    station_context: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    model: str = "qwen3:8b",
    temperature: float = 0.2,
    log_stage: Optional[str] = None,
    log_payload: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
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
        metadata.update(
            {
                "has_query": query is not None,
                "has_raw_data": raw_data is not None,
                "comparison_count": len(comparisons or []),
                "neighbor_count": len(neighbors or []),
                "has_context": station_context is not None,
            }
        )
        answer = "타임 시리즈 분석 요약이 여기에 표시됩니다."

    return {
        "answer": answer,
        "metadata": metadata,
        "log": log_entry,
    }

