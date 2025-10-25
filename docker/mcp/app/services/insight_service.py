import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from itertools import cycle
from typing import Any, Dict, List, Optional
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

from .common import parse_series_values

logger = logging.getLogger(__name__)


__all__ = ["build_insight_payload"]


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
            raw = entry.get("values")
            if isinstance(raw, str) and raw.strip():
                documents.append(raw)
                continue
            floats = parse_series_values(entry)
            documents.append(",".join("nan" if math.isnan(val) else str(val) for val in floats))
        elif isinstance(entry, str):
            documents.append(entry)
        elif np is not None and isinstance(entry, np.ndarray):
            floats = _coerce_float_sequence(entry)
            if floats:
                documents.append(
                    ",".join("nan" if math.isnan(val) else str(val) for val in floats)
                )
        elif isinstance(entry, (list, tuple)):
            floats: List[float] = []
            for val in entry:
                try:
                    floats.append(float(val))
                except (TypeError, ValueError):
                    logger.debug("Skipping non-numeric value in list entry: %s", val)
            if floats:
                documents.append(",".join("nan" if math.isnan(val) else str(val) for val in floats))
        else:
            try:
                value = float(entry)
                documents.append(str(value))
            except (TypeError, ValueError):
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


def _contains_operator(mapping: Dict[Any, Any]) -> bool:
    return any(str(k).startswith("$") for k in mapping.keys())


def _build_condition(key: str, value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if _contains_operator(value):
            return {key: value}
        # Convert nested dict into equality on JSON string representation to avoid ambiguity
        return {key: {"$eq": json.dumps(value, sort_keys=True)}}
    if isinstance(value, list):
        cleaned: List[Any] = [item for item in value if item is not None]
        if not cleaned:
            return None
        return {key: {"$in": cleaned}}
    return {key: {"$eq": value}}


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


def _format_neighbors(chroma_response: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    for idx, metadata, distance, document, embed in zip(
        first_ids,
        first_metadatas,
        first_distances,
        first_documents,
        first_embeddings,
    ):
        cleaned_id = _sanitize_for_json(idx)
        cleaned_metadata = _sanitize_for_json(metadata)
        cleaned_distance = _sanitize_for_json(distance)
        cleaned_document = _sanitize_for_json(document)
        cleaned_embedding = _sanitize_for_json(embed)
        neighbors.append(
            {
                "id": cleaned_id,
                "distance": cleaned_distance,
                "metadata": cleaned_metadata,
                "document": cleaned_document,
                "embedding": cleaned_embedding,
            }
        )
    return neighbors


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
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()

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
        "top_k": top_k,
        "filters": filters or {},
        "element": element,
    }

    generated_embedding: Optional[List[float]] = embedding

    if perform_embedding:
        documents = _compose_documents(values)
        if not documents:
            raise ValueError(
                "values must contain at least one time-series when perform_embedding=True"
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
        generated_embedding = _coerce_float_sequence(first_embedding)
        metadata.update(
            {
                "embedding_source": "generated",
                "input_value_count": len(_normalize_entries(values)),
                "trep_device": resolved_device,
                "trep_model_path": str(weight_path),
                "embedding_dim": len(generated_embedding),
            }
        )
    else:
        if generated_embedding is None:
            raise ValueError(
                "embedding must be provided when perform_embedding=False"
            )
        generated_embedding = _coerce_float_sequence(generated_embedding)
        metadata.update(
            {
                "embedding_source": "provided",
                "input_value_count": len(generated_embedding),
                "embedding_dim": len(generated_embedding),
            }
        )

    neighbors: List[Dict[str, Any]] = []
    if perform_search:
        if generated_embedding is None:
            raise ValueError("embedding vector required for vector search")

        base_url = _compose_base_url(settings.vector_db_host, settings.vector_db_port)
        collection_name = effective_collection
        try:
            collection_id = _resolve_collection_id(base_url, collection_name)
            metadata["collection_id"] = collection_id
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
            response = _query_chromadb(
                base_url,
                collection_id,
                generated_embedding,
                top_k=top_k or settings.vector_db_top_k,
                filters=where_clause,
            )
            neighbors = _format_neighbors(response)
            metadata["returned_neighbors"] = len(neighbors)
        except requests.RequestException as exc:
            logger.error("Failed to query ChromaDB: %s", exc)
            metadata["search_error"] = str(exc)
        except RuntimeError as exc:
            logger.error("ChromaDB error: %s", exc)
            metadata["search_error"] = str(exc)

    return {
        "embedding": _sanitize_for_json(generated_embedding),
        "neighbors": [_sanitize_for_json(item) for item in neighbors],
        "metadata": _sanitize_for_json(metadata),
    }
