import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from datetime import datetime

import numpy as np
import pandas as pd
from fastdtw import fastdtw

from config import Settings
from vendor.modules.NIER.postgres_handler import fetch_data
from internal.analysis_cache import cache_series_payload
from .station_network import StationNetwork

from .common import ensure_sequence, parse_series_values

__all__ = [
    "select_related_stations",
    "build_station_context",
    "sliding_fast_dtw",
    "compute_similarity_metrics",
    "perform_timeseries_analysis",
]


@lru_cache(maxsize=1)
def _get_station_network() -> StationNetwork:
    """Lazily instantiate StationNetwork to avoid repeated pickle loads."""
    return StationNetwork()


def select_related_stations(
    station_id: int,
    element: str,
    *,
    max_related: Optional[int],
) -> List[int]:
    network = _get_station_network()
    related = network.get_related_station(station_id, element) or []
    if max_related is not None:
        return related[:max_related]
    return related


def build_station_context(
    station_id: int,
    element: str,
    selected_related: Sequence[int],
) -> Dict[str, Any]:
    """Assemble lightweight station metadata for downstream prompts."""
    network = _get_station_network()
    all_candidates = network.get_related_station(station_id, element) or []
    return {
        "station_id": station_id,
        "element": element,
        "selected_related": ensure_sequence(selected_related),
        "candidate_related": ensure_sequence(all_candidates),
        "geo": {},
        "stats": {},
    }


def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        # treat as unix timestamp (seconds)
        dt = datetime.fromtimestamp(value)
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError("Timestamp value cannot be empty")

        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ]

        dt = None
        for pattern in patterns:
            try:
                dt = datetime.strptime(candidate, pattern)
                break
            except ValueError:
                continue

        if dt is None:
            try:
                dt = datetime.fromisoformat(candidate)
            except ValueError as exc:
                raise ValueError(
                    "Unsupported timestamp format. Expected ISO-like string."
                ) from exc
    else:
        raise TypeError("Timestamp value must be str, datetime, or unix timestamp")

    return dt


def _handle_missing_values(values: Sequence[float], policy: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    if policy == "mean":
        mean_value = np.nanmean(array) if not np.isnan(array).all() else 0.0
        return np.nan_to_num(array, nan=mean_value)
    if policy == "interpolate":
        series = pd.Series(array)
        interpolated = series.interpolate(method="linear", limit_direction="both")
        return interpolated.fillna(0).to_numpy()

    # Default policy: replace NaN with 0.
    return np.nan_to_num(array, nan=0.0)


def sliding_fast_dtw(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    window_size: int = 6,
    missing_value_policy: str = "zero",
) -> Optional[float]:
    """Compute FastDTW distance using a sliding window strategy."""
    x = _handle_missing_values(x_values, missing_value_policy)
    y = _handle_missing_values(y_values, missing_value_policy)

    n = min(len(x), len(y))
    if n == 0:
        return None

    if n <= window_size or window_size <= 1:
        distance, _ = fastdtw(x[:n], y[:n])
        return float(distance)

    distances: List[float] = []
    step = window_size
    for start in range(0, n - window_size + 1, step):
        x_window = x[start : start + window_size]
        y_window = y[start : start + window_size]
        distance, _ = fastdtw(x_window, y_window)
        distances.append(float(distance))

    if not distances:
        return None
    return float(np.mean(distances))


def _summarize_distances(distances: Sequence[Optional[float]]) -> Optional[Dict[str, float]]:
    valid = [d for d in distances if d is not None and not math.isnan(d)]
    if not valid:
        return None

    array = np.asarray(valid, dtype=float)
    summary = {
        "count": float(len(valid)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)) if len(valid) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q1": float(np.quantile(array, 0.25)) if len(valid) > 1 else float(array[0]),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)) if len(valid) > 1 else float(array[0]),
    }
    return summary


def _assign_similarity_labels(
    results: List[Dict[str, Any]],
    summary: Optional[Dict[str, float]],
) -> None:
    if not summary:
        for item in results:
            item["label"] = "insufficient_data" if item["distance"] is None else "undetermined"
        return

    q1 = summary["q1"]
    q3 = summary["q3"]
    for item in results:
        distance = item["distance"]
        if distance is None or math.isnan(distance):
            item["label"] = "insufficient_data"
            continue
        if distance <= q1:
            item["label"] = "high_similarity"
        elif distance <= q3:
            item["label"] = "moderate_similarity"
        else:
            item["label"] = "low_similarity"


def _safe_station_id(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def compute_similarity_metrics(
    original_series: Dict[str, Any],
    related_series: Sequence[Dict[str, Any]],
    *,
    comparison_type: str,
    window_size: int,
    missing_value_policy: str,
) -> Dict[str, Any]:
    base_values = parse_series_values(original_series)
    if not base_values:
        return {
            "results": [],
            "summary": None,
            "metadata": {
                "reason": "original_series_empty",
                "window_size": window_size,
                "missing_value_policy": missing_value_policy,
            },
        }

    comparisons: List[Dict[str, Any]] = []
    distances: List[Optional[float]] = []

    for item in related_series:
        target_station = _safe_station_id(item.get("region") or item.get("station_id"))
        related_values = parse_series_values(item)
        if not related_values:
            comparisons.append(
                {
                    "station_id": target_station,
                    "distance": None,
                    "label": "insufficient_data",
                    "comparison_type": comparison_type,
                    "series_length": 0,
                }
            )
            distances.append(None)
            continue

        distance = sliding_fast_dtw(
            base_values,
            related_values,
            window_size=window_size,
            missing_value_policy=missing_value_policy,
        )
        distances.append(distance)
        comparisons.append(
            {
                "station_id": target_station,
                "distance": float(distance) if distance is not None else None,
                "label": "pending",
                "comparison_type": comparison_type,
                "series_length": len(related_values),
            }
        )

    summary = _summarize_distances(distances)
    _assign_similarity_labels(comparisons, summary)

    return {
        "results": comparisons,
        "summary": summary,
        "metadata": {
            "window_size": window_size,
            "missing_value_policy": missing_value_policy,
        },
    }


def perform_timeseries_analysis(
    settings: Settings,
    *,
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
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
    element = element.upper()
    start_dt = _normalize_timestamp(start_time)
    end_dt = _normalize_timestamp(end_time)

    if start_dt > end_dt:
        raise ValueError("start_time must be earlier than end_time")

    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    double_sequence = (
        settings.double_the_sequence if double_sequence is None else double_sequence
    )
    additional_days = (
        settings.additional_days if additional_days is None else additional_days
    )

    query = {
        "type": "time_series",
        "region": int(station_id),
        "element": element,
        "start_time": start_time,
        "end_time": end_time,
    }

    original_data = fetch_data(
        settings.postgres_user,
        settings.postgres_password,
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
        double_sequence,
        additional_days,
        query,
        settings.db_csv_path,
    )

    related_results: List[Dict[str, Any]] = []
    related_errors: List[Dict[str, Any]] = []
    related_station_ids: List[int] = []

    if include_related:
        effective_max = settings.max_related if max_related is None else max_related
        related_station_ids = select_related_stations(
            station_id=int(station_id),
            element=element,
            max_related=effective_max,
        )
        for related_station in related_station_ids:
            try:
                station_query = {
                    **query,
                    "region": int(related_station),
                }
                station_data = fetch_data(
                    settings.postgres_user,
                    settings.postgres_password,
                    settings.postgres_host,
                    settings.postgres_port,
                    settings.postgres_db,
                    double_sequence,
                    additional_days,
                    station_query,
                    settings.db_csv_path,
                )
                related_results.append(station_data)
            except Exception as exc:  # pragma: no cover - surface to caller
                related_errors.append(
                    {
                        "station_id": int(related_station),
                        "error": str(exc),
                    }
                )

    context_payload: Optional[Dict[str, Any]] = None
    if include_context:
        context_payload = build_station_context(
            station_id=int(station_id),
            element=element,
            selected_related=related_station_ids,
        )

    similarity_payload: List[Dict[str, Any]] = []
    comparison_summary: Optional[Dict[str, Any]] = None
    comparison_metadata: Optional[Dict[str, Any]] = None
    if compute_similarity and related_results:
        similarity_output = compute_similarity_metrics(
            original_series=original_data,
            related_series=related_results,
            comparison_type=comparison_type,
            window_size=window_size,
            missing_value_policy=missing_value_policy,
        )
        similarity_payload = similarity_output["results"]
        comparison_summary = similarity_output["summary"]
        comparison_metadata = similarity_output["metadata"]

    original_values_raw = original_data.get("values", "")
    try:
        original_value_count = len(parse_series_values({"values": original_values_raw}))
    except Exception:  # pragma: no cover - defensive
        original_value_count = 0
    value_preview = []
    if isinstance(original_values_raw, str) and original_values_raw:
        value_preview = [
            token
            for token in (original_values_raw.split(",")[:6])
            if token not in ("", None)
        ]

    cache_key = cache_series_payload(
        {
            "region": original_data.get("region"),
            "element": original_data.get("element"),
            "start_time": original_data.get("start_time"),
            "end_time": original_data.get("end_time"),
            "values": original_values_raw,
        }
    )
    original_data["cache_key"] = cache_key
    original_data["values"] = None
    original_data["value_preview"] = value_preview
    original_data["value_count"] = original_value_count

    metadata = {
        "double_sequence": double_sequence,
        "additional_days": additional_days,
        "include_related": include_related,
        "include_context": include_context,
        "compute_similarity": compute_similarity,
        "comparison_type": comparison_type,
        "window_size": window_size,
        "missing_value_policy": missing_value_policy,
        "comparison_metadata": comparison_metadata,
        "analysis_cache_key": cache_key,
        "analysis_value_count": original_value_count,
    }

    return {
        "query": query,
        "original": original_data,
        "related_station_ids": ensure_sequence(related_station_ids),
        "related": related_results,
        "related_errors": related_errors,
        "context": context_payload,
        "comparisons": similarity_payload,
        "comparison_summary": comparison_summary,
        "metadata": metadata,
    }
