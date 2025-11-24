import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from itertools import cycle
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

import requests

try:  # Optional dependency, required when generating embeddings
    import torch  # type: ignore
except ImportError:  # pragma: no cover - surface clear message later
    torch = None  # type: ignore

try:  # NumPy is preferred for handling upstream vector formats
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from config import get_settings
from vendor.modules.NIER.chroma_trep import TRepEmbedding  # type: ignore
from internal.analysis_cache import load_series_payload, cache_series_payload

from .common import parse_series_values

logger = logging.getLogger(__name__)


__all__ = ["build_insight_payload"]

_FILTER_KEY_ALIASES: Dict[str, str] = {
    "station_id": "region",
    "station": "region",
    "region": "region",
    "element": "element",
    "class": "class",
    "start_time": "original_start",
    "end_time": "original_end",
    "original_start": "original_start",
    "original_end": "original_end",
    "start": "original_start",
    "end": "original_end",
    "period_start": "original_start",
    "period_end": "original_end",
    "from_time": "original_start",
    "to_time": "original_end",
    "begin_time": "original_start",
    "finish_time": "original_end",
}


def _sanitize_device(device: str) -> str:
    device = (device or "").strip() or "cpu"
    if device.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to cpu.")
            return "cpu"
        if ":" in device:
            try:
                index = int(device.split(":", 1)[1])
            except ValueError:
                logger.warning("Invalid CUDA device '%s'. Using cuda:0", device)
                return "cuda:0"
            if index >= torch.cuda.device_count():
                logger.warning(
                    "CUDA device index %s out of range. Using cuda:0", index
                )
                return "cuda:0"
    return device


@lru_cache(maxsize=1)
def _resolve_device_pool() -> List[str]:
    env_value = os.getenv("NIER_TREP_DEFAULT_DEVICE", "").strip()
    candidates: List[str] = []
    if env_value:
        tokens = [token.strip() for token in env_value.split(",") if token.strip()]
        candidates.extend(tokens)
    elif torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        candidates.extend([f"cuda:{idx}" for idx in range(gpu_count)])
    if not candidates:
        candidates.append("cpu")

    sanitized: List[str] = []
    for candidate in candidates:
        sanitized_device = _sanitize_device(candidate)
        if sanitized_device not in sanitized:
            sanitized.append(sanitized_device)

    if not sanitized:
        sanitized = ["cpu"]

    logger.info("T-Rep device pool resolved to: %s", sanitized)
    return sanitized


_DEVICE_CYCLE = None


def _next_device() -> str:
    global _DEVICE_CYCLE
    if _DEVICE_CYCLE is None:
        devices = _resolve_device_pool()
        _DEVICE_CYCLE = cycle(devices)
    return next(_DEVICE_CYCLE)


def _resolve_weight_path(model_dir: str, element: str) -> Path:
    element = element.upper()
    base_dir = Path(model_dir)
    candidate = base_dir / f"model_{element}.pt"
    if candidate.exists():
        return candidate
    generic = base_dir / "model.pt"
    if generic.exists():
        logger.warning(
            "Specific T-Rep weight for %s not found. Using generic model at %s.",
            element,
            generic,
        )
        return generic
    raise FileNotFoundError(
        f"T-Rep weight file not found for element '{element}' in {model_dir}"
    )


def _normalize_entries(values: Optional[Any]) -> List[Any]:
    if values is None:
        return []
    if np is not None and isinstance(values, np.ndarray):
        return _normalize_entries(values.tolist())
    if isinstance(values, dict):
        return [values]
    if isinstance(values, (list, tuple)):
        sequence = list(values)
        if not sequence:
            return []
        if all(
            not isinstance(item, (dict, list, tuple))
            and not (np is not None and isinstance(item, np.ndarray))
            for item in sequence
        ):
            return [sequence]
        return sequence
    return [values]


def _compose_documents(values: Optional[Any]) -> List[str]:
    documents: List[str] = []
    for entry in _normalize_entries(values):
        if isinstance(entry, dict):
            cache_key = (
                entry.get("cache_key")
                or entry.get("analysis_cache_key")
                or entry.get("series_cache_key")
            )
            raw = entry.get("values")
            if cache_key:
                try:
                    cached_payload = load_series_payload(str(cache_key))
                except KeyError as exc:  # pragma: no cover - surface to caller
                    raise RuntimeError(
                        f"Cached time-series payload not found for key {cache_key}"
                    ) from exc
                raw = cached_payload.get("values") or raw
            elif raw:
                # Direct values usage allowed
                pass
            else:
                raise ValueError(
                    "cache_key가 누락되었습니다. 먼저 timeseries_analysis를 호출해 cache_key를 확보하세요."
                )

            floats = parse_series_values({"values": raw})
            documents.append(",".join("nan" if math.isnan(val) else str(val) for val in floats))
        elif isinstance(entry, str):
            raise ValueError(
                "직접 문자열 시계열을 입력할 수 없습니다. cache_key를 이용해 주세요."
            )
        elif np is not None and isinstance(entry, np.ndarray):
            raise ValueError(
                "numpy.ndarray 입력은 지원되지 않습니다. cache_key를 이용해 주세요."
            )
        elif isinstance(entry, (list, tuple)):
            raise ValueError(
                "리스트 기반 values 입력은 허용되지 않습니다. cache_key를 사용하세요."
            )
        else:
            logger.debug("Skipping unsupported value type in documents: %s", entry)
    return documents


@lru_cache(maxsize=16)
def _load_trep_embedding(
    element: str,
    weight_path: str,
    device: str,
    encoding_window: str,
    time_embedding: Optional[str],
) -> "TRepEmbedding":
    if torch is None:  # pragma: no cover - configuration error
        raise RuntimeError("PyTorch is required for T-Rep embeddings")
    logger.info(
        "Loading T-Rep weights for %s from %s on %s",
        element,
        weight_path,
        device,
    )
    return TRepEmbedding(
        weight_path=weight_path,
        device=device,
        encoding_window=encoding_window,
        time_embedding=time_embedding,
    )


def _compose_base_url(host: str, port: str) -> str:
    host = (host or "http://localhost").rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    if port and f":{port}" not in host.split("//", 1)[-1]:
        host = f"{host}:{port}"
    return host


def _normalize_collection_name(requested: Optional[Any], default: str) -> str:
    if not requested:
        return default
    if not isinstance(requested, str):
        requested = str(requested)
    normalized = requested.strip()
    if not normalized:
        return default

    alias_map = {
        "trep": default,
        "trep_collection": default,
        "time_series_collection": default,
        "time_series_collection_trep": default,
    }
    lowered = normalized.lower()
    if lowered in alias_map:
        alias_target = alias_map[lowered]
        logger.debug(
            "Collection alias '%s' resolved to '%s'",
            normalized,
            alias_target,
        )
        return alias_target
    return normalized


def _normalize_timestamp_string(value: Any, *, is_end: bool = False) -> Any:
    if value is None:
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return candidate
        candidate = candidate.replace("T", " ")
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]
        for pattern in patterns:
            try:
                dt = datetime.strptime(candidate, pattern)
                if pattern == "%Y-%m-%d":
                    dt = dt.replace(
                        hour=23 if is_end else 0,
                        minute=59 if is_end else 0,
                        second=59 if is_end else 0,
                    )
                elif pattern == "%Y-%m-%d %H:%M":
                    dt = dt.replace(second=59 if is_end else 0)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return candidate
    return value


def _normalize_filter_payload(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not filters:
        return filters
    normalized: Dict[str, Any] = {}
    time_keys = {
        "start_time",
        "original_start",
        "period_start",
        "from_time",
        "begin_time",
    }
    end_time_keys = {
        "end_time",
        "original_end",
        "period_end",
        "to_time",
        "finish_time",
    }
    for key, value in filters.items():
        is_end_key = key in end_time_keys or key.endswith("_end")
        is_start_key = key in time_keys or key.endswith("_start")
        if isinstance(value, dict):
            normalized[key] = _normalize_filter_payload(value)  # type: ignore[assignment]
            continue
        if isinstance(value, list):
            normalized_list: List[Any] = []
            for item in value:
                if is_end_key or is_start_key:
                    normalized_list.append(
                        _normalize_timestamp_string(
                            item,
                            is_end=is_end_key and not is_start_key,
                        )
                    )
                elif isinstance(item, dict):
                    normalized_list.append(_normalize_filter_payload(item))
                else:
                    normalized_list.append(item)
            normalized[key] = normalized_list
            continue
        if is_end_key:
            normalized[key] = _normalize_timestamp_string(value, is_end=True)
        elif is_start_key:
            normalized[key] = _normalize_timestamp_string(value, is_end=False)
        else:
            normalized[key] = value
    return normalized


def _resolve_filter_key(raw_key: Any) -> str:
    key = str(raw_key).strip()
    alias = _FILTER_KEY_ALIASES.get(key.lower())
    return alias or key


def _contains_operator(mapping: Dict[Any, Any]) -> bool:
    return any(str(k).startswith("$") for k in mapping.keys())


def _build_condition(key: str, value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    resolved_key = _resolve_filter_key(key)
    if isinstance(value, dict):
        if _contains_operator(value):
            return {resolved_key: value}
        # Convert nested dict into equality on JSON string representation to avoid ambiguity
        return {resolved_key: {"$eq": json.dumps(value, sort_keys=True)}}
    if isinstance(value, list):
        cleaned: List[Any] = [item for item in value if item is not None]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return {resolved_key: {"$eq": cleaned[0]}}
        return {resolved_key: {"$in": cleaned}}
    return {resolved_key: {"$eq": value}}


def _to_chroma_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not filters:
        return None
    if _contains_operator(filters):
        return filters

    conditions: List[Dict[str, Any]] = []
    for key, value in filters.items():
        condition = _build_condition(key, value)
        if condition:
            conditions.append(condition)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@lru_cache(maxsize=8)
def _resolve_collection_id(base_url: str, collection_name: str) -> str:
    url = f"{base_url}/api/v1/collections"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    collections = response.json()
    for collection in collections:
        if collection.get("name") == collection_name:
            return collection.get("id")
    raise RuntimeError(f"Collection '{collection_name}' not found on {base_url}")


def _query_chromadb(
    base_url: str,
    collection_id: str,
    embedding: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances", "embeddings"],
    }
    if filters:
        payload["where"] = filters

    url = f"{base_url}/api/v1/collections/{collection_id}/query"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def _extract_time_range(metadata: Dict[str, Any]) -> Optional[tuple[datetime, datetime]]:
    start_keys = ("original_start", "start_time", "start")
    end_keys = ("original_end", "end_time", "end")
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    for key in start_keys:
        candidate = metadata.get(key)
        start_dt = _parse_datetime(candidate) if candidate else None
        if start_dt:
            break

    for key in end_keys:
        candidate = metadata.get(key)
        end_dt = _parse_datetime(candidate) if candidate else None
        if end_dt:
            break

    if start_dt and end_dt:
        return start_dt, end_dt
    return None


def _ranges_overlap(first: tuple[datetime, datetime], second: tuple[datetime, datetime]) -> bool:
    start_a, end_a = first
    start_b, end_b = second
    return not (end_a < start_b or end_b < start_a)


def _format_neighbors(
    chroma_response: Dict[str, Any],
    *,
    target_element: Optional[str],
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    ids = chroma_response.get("ids", [[]])
    metadatas = chroma_response.get("metadatas", [[]])
    distances = chroma_response.get("distances", [[]])
    documents = chroma_response.get("documents", [[]])
    embeddings = chroma_response.get("embeddings", [[]])

    if not ids or not isinstance(ids, list):
        return []

    first_ids = ids[0] if ids else []
    first_metadatas = metadatas[0] if metadatas else []
    first_distances = distances[0] if distances else []
    first_documents = documents[0] if documents else []
    first_embeddings = embeddings[0] if embeddings else []

    neighbors: List[Dict[str, Any]] = []
    seen_ranges: List[tuple[Any, str, tuple[datetime, datetime]]] = []
    normalized_element = target_element.upper() if target_element else None

    for idx, metadata, distance, document, embed in zip(
        first_ids,
        first_metadatas,
        first_distances,
        first_documents,
        first_embeddings,
    ):
        sanitized_metadata = _sanitize_for_json(metadata)
        if not isinstance(sanitized_metadata, dict):
            continue

        element_value = str(sanitized_metadata.get("element", "")).upper()
        if normalized_element and element_value != normalized_element:
            continue

        region_value = (
            sanitized_metadata.get("region")
            or sanitized_metadata.get("station")
            or sanitized_metadata.get("station_id")
        )
        time_range = _extract_time_range(sanitized_metadata)
        if time_range and region_value is not None:
            if any(
                existing_region == region_value
                and existing_element == element_value
                and _ranges_overlap(time_range, existing_range)
                for existing_region, existing_element, existing_range in seen_ranges
            ):
                continue
            seen_ranges.append((region_value, element_value, time_range))

        sanitized_metadata.pop("values", None)
        sanitized_metadata.pop("document", None)

        neighbor_payload = {
            "id": _sanitize_for_json(idx),
            "distance": _sanitize_for_json(distance),
            "metadata": sanitized_metadata,
        }
        neighbors.append(neighbor_payload)

        if len(neighbors) >= max_neighbors:
            break
    return neighbors


def _class_label(value: Any) -> str:
    if str(value) == "0":
        return "정상"
    if str(value) == "3":
        return "이상"
    return "미분류"


def _similarity_label(distance: float) -> str:
    if distance < 2.0:
        return "유사"
    if distance < 5.0:
        return "다름"
    return "매우 다름"


def _summarize_neighbors(neighbors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    abnormal_weight = 0.0
    normal_weight = 0.0
    total_weight = 0.0
    summarized_neighbors: List[Dict[str, Any]] = []

    for neighbor in neighbors:
        distance = neighbor.get("distance")
        if distance is None:
            continue
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            continue

        weight = 1.0 / (distance_value + 1e-6)
        total_weight += weight

        metadata = neighbor.get("metadata") or {}
        classification = metadata.get("class")
        if str(classification) == "0":
            normal_weight += weight
        elif str(classification) == "3":
            abnormal_weight += weight

        summarized_neighbors.append(
            {
                "station_id": metadata.get("region")
                or metadata.get("station")
                or metadata.get("station_id"),
                "element": metadata.get("element"),
                "class_label": _class_label(classification),
                "similarity": _similarity_label(distance_value),
                "original_start": metadata.get("original_start"),
                "original_end": metadata.get("original_end"),
            }
        )

    abnormal_probability = (
        (abnormal_weight / total_weight) * 100.0 if total_weight > 0 else 0.0
    )

    return {
        "abnormal_probability": abnormal_probability,
        "neighbors": summarized_neighbors,
    }


def _coerce_float_sequence(sequence: Any) -> List[float]:
    if sequence is None:
        return []
    if np is not None and isinstance(sequence, (np.ndarray, np.generic)):
        return _coerce_float_sequence(sequence.tolist())  # type: ignore[attr-defined]
    if torch is not None and isinstance(sequence, torch.Tensor):
        return _coerce_float_sequence(sequence.detach().cpu().tolist())
    if hasattr(sequence, "tolist") and not isinstance(sequence, (list, tuple)):
        return _coerce_float_sequence(sequence.tolist())
    if isinstance(sequence, (list, tuple)):
        result: List[float] = []
        for item in sequence:
            try:
                value = float(item)
            except (TypeError, ValueError):
                logger.debug("Skipping non-numeric value in embedding payload: %s", item)
                continue
            if math.isnan(value):
                value = float("nan")
            result.append(value)
        return result
    try:
        value = float(sequence)
    except (TypeError, ValueError) as exc:
        raise TypeError("Expected a numeric sequence for embedding") from exc
    if math.isnan(value):
        value = float("nan")
    return [value]


def _sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if np is not None:
        if isinstance(value, np.generic):
            return _sanitize_for_json(value.item())
        if isinstance(value, np.ndarray):
            return [_sanitize_for_json(item) for item in value.tolist()]
    if torch is not None and isinstance(value, torch.Tensor):
        return _sanitize_for_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _sanitize_for_json(value.tolist())
    try:
        return _sanitize_for_json(float(value))
    except (TypeError, ValueError):
        return str(value)


def build_insight_payload(
    values: Optional[Any] = None,
    *,
    element: Optional[str] = None,
    collection: Optional[str] = None,
    perform_embedding: bool = True,
    perform_search: bool = True,
    embedding: Optional[List[float]] = None,
    device: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    target_top_k = max(settings.vector_db_top_k, 1)
    prefilter_top_k = max(settings.vector_db_prefilter_top_k, target_top_k)

    if perform_embedding and torch is None:
        raise RuntimeError("PyTorch is required for T-Rep embeddings. Install torch.")

    effective_collection = _normalize_collection_name(
        collection,
        settings.vector_collection,
    )

    metadata: Dict[str, Any] = {
        "perform_embedding": perform_embedding,
        "perform_search": perform_search,
        "collection": effective_collection,
        "vector_db_host": settings.vector_db_host,
        "vector_db_port": settings.vector_db_port,
        "top_k": target_top_k,
        "prefilter_top_k": prefilter_top_k,
        "filters": filters or {},
        "element": element,
    }

    embedding_vector: Optional[List[float]] = embedding
    embedding_cache_key: Optional[str] = None

    if perform_embedding:
        if not values:
            raise ValueError(
                "perform_embedding=True인 경우 values=[{'cache_key': ...}] 형태로 timeseries_analysis의 cache_key를 전달해야 합니다."
            )
        documents = _compose_documents(values)
        if not documents:
            raise ValueError(
                "cache_key를 통해 조회한 시계열이 비어 있습니다. timeseries_analysis 응답을 확인해 주세요."
            )

        target_element = (element or "SO2").upper()
        weight_path = _resolve_weight_path(settings.trep_model_dir, target_element)
        resolved_device = _sanitize_device(device or _next_device())
        encoding_window = os.getenv("NIER_TREP_ENCODING_WINDOW", "full_series")
        time_embedding = os.getenv("NIER_TREP_TIME_EMBEDDING", "t2v_sin")

        embedding_fn = _load_trep_embedding(
            target_element,
            str(weight_path),
            resolved_device,
            encoding_window,
            time_embedding,
        )

        trep_embeddings = embedding_fn(documents)
        if not trep_embeddings:
            raise RuntimeError("T-Rep embedding returned no results")

        first_embedding = trep_embeddings[0]
        embedding_vector = _coerce_float_sequence(first_embedding)
        embedding_cache_key = cache_series_payload(
            {
                "type": "embedding",
                "element": target_element,
                "vector": embedding_vector,
            }
        )
        metadata.update(
            {
                "embedding_source": "generated",
                "input_value_count": len(_normalize_entries(values)),
                "trep_device": resolved_device,
                "trep_model_path": str(weight_path),
                "embedding_dim": len(embedding_vector),
            }
        )
    else:
        if isinstance(embedding_vector, str):
            cached_payload = load_series_payload(embedding_vector)
            embedding_cache_key = embedding_vector
            embedding_vector = _coerce_float_sequence(
                cached_payload.get("vector")
                or cached_payload.get("embedding")
            )
        if embedding_vector is None:
            raise ValueError(
                "embedding must be provided when perform_embedding=False"
            )
        embedding_vector = _coerce_float_sequence(embedding_vector)
        if embedding_cache_key is None:
            embedding_cache_key = cache_series_payload(
                {
                    "type": "embedding",
                    "element": (element or "UNKNOWN").upper(),
                    "vector": embedding_vector,
                }
            )
        metadata.update(
            {
                "embedding_source": "provided",
                "input_value_count": len(embedding_vector),
                "embedding_dim": len(embedding_vector),
            }
        )

    neighbors: List[Dict[str, Any]] = []
    if perform_search:
        if embedding_vector is None:
            raise ValueError("embedding vector required for vector search")

        base_url = _compose_base_url(settings.vector_db_host, settings.vector_db_port)
        collection_name = effective_collection
        try:
            collection_id = _resolve_collection_id(base_url, collection_name)
            metadata["collection_id"] = collection_id
            element_upper = element.upper() if element else None
            where_filters = dict(filters or {})
            if element:
                where_filters.setdefault("element", element.upper())
            sanitized_filters = _sanitize_for_json(where_filters) or None
            if sanitized_filters is not None and not isinstance(sanitized_filters, dict):
                sanitized_filters = {"value": sanitized_filters}
            normalized_filters = (
                _normalize_filter_payload(sanitized_filters)
                if isinstance(sanitized_filters, dict)
                else sanitized_filters
            )
            where_clause = (
                _to_chroma_where(normalized_filters)
                if isinstance(normalized_filters, dict)
                else None
            )
            metadata["filters"] = normalized_filters or {}
            metadata["where_clause"] = where_clause or {}

            def _run_query(active_filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
                response = _query_chromadb(
                    base_url,
                    collection_id,
                    embedding_vector,
                    top_k=prefilter_top_k,
                    filters=active_filters,
                )
                return _format_neighbors(
                    response,
                    target_element=element_upper,
                    max_neighbors=target_top_k,
                )

            primary_error: Optional[str] = None
            try:
                neighbors = _run_query(where_clause)
                metadata["returned_neighbors"] = len(neighbors)
            except requests.RequestException as exc:
                primary_error = str(exc)
                metadata["search_error"] = primary_error
                neighbors = []
            except RuntimeError as exc:
                primary_error = str(exc)
                metadata["search_error"] = primary_error
                neighbors = []

            if (not neighbors or primary_error) and isinstance(normalized_filters, dict):
                element_filter = normalized_filters.get("element")
                if isinstance(element_filter, dict) and "$eq" in element_filter:
                    element_filter = element_filter["$eq"]
                fallback_source: Dict[str, Any] = {}
                if element_filter:
                    fallback_source["element"] = element_filter
                fallback_where = _to_chroma_where(fallback_source) if fallback_source else None
                if fallback_where != where_clause:
                    metadata["fallback_filters"] = fallback_source
                    metadata["fallback_where_clause"] = fallback_where or {}
                    try:
                        fallback_neighbors = _run_query(fallback_where)
                        if fallback_neighbors:
                            neighbors = fallback_neighbors
                            metadata["returned_neighbors"] = len(neighbors)
                            if primary_error:
                                metadata["search_error_primary"] = primary_error
                                metadata.pop("search_error", None)
                    except requests.RequestException as exc:
                        metadata["fallback_error"] = str(exc)
                    except RuntimeError as exc:
                        metadata["fallback_error"] = str(exc)
        except requests.RequestException as exc:
            logger.error("Failed to query ChromaDB: %s", exc)
            metadata["search_error"] = str(exc)
        except RuntimeError as exc:
            logger.error("ChromaDB error: %s", exc)
            metadata["search_error"] = str(exc)

    if embedding_cache_key is None and embedding_vector is not None:
        embedding_cache_key = cache_series_payload(
            {
                "type": "embedding",
                "element": (element or "UNKNOWN").upper(),
                "vector": embedding_vector,
            }
        )

    summary = _summarize_neighbors(neighbors)

    return {
        "abnormal_probability": summary["abnormal_probability"],
        "neighbors": summary["neighbors"],
    }
