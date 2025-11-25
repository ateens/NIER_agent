import json
import logging
import math
import os
import pickle
import threading
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from fastdtw import fastdtw

from config import get_settings
from vendor.modules.NIER.postgres_handler import fetch_data
from vendor.modules.NIER.chroma_trep import TRepEmbedding

# =============================================================================
# Global Constants & Defaults
# =============================================================================

WINDOW_SIZE = 6
COMPARISON_TYPE = "dtw"
MISSING_VALUE_POLICY = "zero"
DOUBLE_SEQUENCE = None  # Use settings default
ADDITIONAL_DAYS = None  # Use settings default
INCLUDE_RELATED = True
COMPUTE_SIMILARITY = True
MAX_RELATED = None
RESPONSE_TYPE = "analysis"

_Z_SCORE_COMMENTS = {
    "평균 유사도": "와의 유사도가 평균 범위 내에 있습니다.",
    "약간 낮은 유사도": "와의 유사도가 약간 낮습니다.",
    "낮은 유사도": "와의 유사도가 평균보다 낮으므로 이상 징후 의심이 됩니다.",
    "매우 낮은 유사도": "와의 유사도가 평균보다 크게 낮아 베이스라인 이상 징후가 강하게 의심됩니다.",
    "데이터 부족": "에 대한 비교 데이터를 확보하지 못했습니다.",
}

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Classes & Functions (Consolidated)
# =============================================================================

class StationNetwork:
    """Lightweight loader for station similarity groups."""

    def __init__(self, similarity_path: Optional[str] = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        default_path = base_dir / "vendor" / "assets" / "similarity_results_v2.pkl"

        legacy_path = None
        # Fallback search
        for ancestor in base_dir.parents:
            candidate = ancestor / "vendor" / "assets" / "similarity_results_v2.pkl"
            if candidate.exists():
                legacy_path = candidate
                break

        env_path = os.getenv("NIER_STATION_SIMILARITY_PATH")

        candidates = [
            Path(p)
            for p in [similarity_path, env_path, legacy_path, default_path]
            if p
        ]

        self._groups: Dict[int, Dict[str, Dict[int, Dict[str, float]]]] = {}
        for path in candidates:
            if path.exists():
                self._groups = self._load_pickle(path)
                break

        if not self._groups:
            # Fallback or empty if not found, rather than crashing if file missing in dev
            logger.warning("Could not load station similarity data.")

    @staticmethod
    def _load_pickle(path: Path) -> Dict[int, Dict[str, Dict[int, Dict[str, float]]]]:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def get_related_station(self, station_id: int, element: str) -> List[int]:
        station_id = int(station_id)
        element = element.upper()
        if int(station_id) not in self._groups:
            return []
        element_map = self._groups.get(station_id, {})
        if element not in element_map:
            return []
        return list(element_map[element].keys())

    def get_similarity_stats(
        self, station_a: int, station_b: int, element: str, window_size: int
    ) -> Optional[Dict[str, float]]:
        try:
            station_a = int(station_a)
            station_b = int(station_b)
        except (TypeError, ValueError):
            return None

        element = element.upper()
        window_suffix = f"{int(window_size)}h"
        avg_key = f"avg_dist_{window_suffix}"
        std_key = f"sd_{window_suffix}"

        station_payload = self._groups.get(station_a, {})
        element_payload = station_payload.get(element)
        if not element_payload:
            return None

        pair_stats = element_payload.get(station_b)
        if not pair_stats:
            return None

        if avg_key not in pair_stats or std_key not in pair_stats:
            return None

        return {
            "baseline_mean": float(pair_stats[avg_key]),
            "baseline_std": float(pair_stats[std_key]),
        }

@lru_cache(maxsize=1)
def _get_station_network() -> StationNetwork:
    return StationNetwork()

def _to_native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value

def _ensure_sequence(values: Optional[Sequence[Any]]) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, (tuple, set)):
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, np.generic):
        return [values.item()]
    if isinstance(values, str):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]

def _parse_series_values(series: Optional[Dict[str, Any]]) -> List[float]:
    if not series:
        return []
    raw_values = series.get("values")
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        tokens = [token.strip() for token in raw_values.split(",")]
    else:
        tokens = [token for token in _ensure_sequence(raw_values)]
    parsed: List[float] = []
    for token in tokens:
        if token in ("", None):
            continue
        try:
            value = float(token)
        except (TypeError, ValueError):
            continue
        if math.isclose(value, 999999.0):
            value = math.nan
        parsed.append(value)
    return parsed

def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        candidate = value.strip()
        patterns = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
            "%Y-%m-%d", "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M", "%Y/%m/%d"
        ]
        for pattern in patterns:
            try:
                return datetime.strptime(candidate, pattern)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp format: {value}")

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
    return np.nan_to_num(array, nan=0.0)

def _sliding_fast_dtw(
    x_values: Sequence[float],
    y_values: Sequence[float],
    window_size: int = 6,
    missing_value_policy: str = "zero",
) -> Optional[float]:
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

def _classify_z_score(z_score: Optional[float]) -> str:
    if z_score is None or (isinstance(z_score, float) and math.isnan(z_score)):
        return "데이터 부족"
    if z_score < 1.0:
        return "평균 유사도"
    if z_score < 2.0:
        return "약간 낮은 유사도"
    if z_score < 3.0:
        return "낮은 유사도"
    return "매우 낮은 유사도"

def _compute_similarity_metrics(
    original_series: Dict[str, Any],
    related_series: Sequence[Dict[str, Any]],
    comparison_type: str,
    window_size: int,
    missing_value_policy: str,
) -> Dict[str, Any]:
    base_values = _parse_series_values(original_series)
    if not base_values:
        return {"results": [], "summary": None}

    comparisons: List[Dict[str, Any]] = []
    distances: List[Optional[float]] = []
    base_station = _to_native_scalar(original_series.get("region"))
    element = (original_series.get("element") or "").upper()
    station_network = _get_station_network()

    for item in related_series:
        target_station = _to_native_scalar(item.get("region") or item.get("station_id"))
        related_values = _parse_series_values(item)
        if not related_values:
            distances.append(None)
            continue

        distance = _sliding_fast_dtw(
            base_values,
            related_values,
            window_size=window_size,
            missing_value_policy=missing_value_policy,
        )
        
        baseline_mean = None
        baseline_std = None
        z_score = None
        confidence = None

        if distance is not None and base_station is not None and target_station is not None:
            stats = station_network.get_similarity_stats(
                station_a=base_station,
                station_b=target_station,
                element=element,
                window_size=window_size,
            )
            if stats:
                baseline_mean = stats.get("baseline_mean")
                baseline_std = stats.get("baseline_std")
                if baseline_mean is not None and baseline_std is not None and not math.isclose(baseline_std, 0.0):
                    z_score = float((distance - baseline_mean) / baseline_std)
                    confidence = max(0.0, 1.0 - abs(z_score) / 3.0)

        distances.append(distance)
        comparisons.append({
            "station_id": target_station,
            "distance": float(distance) if distance is not None else None,
            "z_score": z_score,
            "confidence": confidence,
            "z_score_range": _classify_z_score(z_score)
        })

    return {"results": comparisons}

# =============================================================================
# Insight Retrieval Helpers
# =============================================================================

def _resolve_device_pool() -> List[str]:
    # Simplified device resolution
    import torch
    if torch.cuda.is_available():
        return ["cuda:0"]
    return ["cpu"]

def _resolve_weight_path(model_dir: str, element: str) -> Path:
    element = element.upper()
    base_dir = Path(model_dir)
    candidate = base_dir / f"model_{element}.pt"
    if candidate.exists():
        return candidate
    return base_dir / "model.pt"

def _coerce_float_sequence(sequence: Any) -> List[float]:
    if sequence is None:
        return []
    
    # Handle numpy
    if isinstance(sequence, (np.ndarray, np.generic)):
        return _coerce_float_sequence(sequence.tolist())
    
    # Handle torch (if available/loaded)
    if hasattr(sequence, "detach"): # Duck typing for torch.Tensor
        try:
            return _coerce_float_sequence(sequence.detach().cpu().tolist())
        except Exception:
            pass
            
    if hasattr(sequence, "tolist"):
        return _coerce_float_sequence(sequence.tolist())

    if isinstance(sequence, (list, tuple)):
        return [float(x) for x in sequence if not math.isnan(float(x))]
        
    return []

def _resolve_collection_id(base_url: str, collection_name: str) -> str:
    url = f"{base_url}/api/v1/collections"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    for collection in response.json():
        if collection.get("name") == collection_name:
            return collection.get("id")
    raise RuntimeError(f"Collection '{collection_name}' not found")

def _query_chromadb(
    base_url: str,
    collection_id: str,
    embedding: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances", "embeddings"],
    }
    if filters:
        payload["where"] = filters
    
    # Detailed logging
    embed_len = len(embedding)
    embed_sample = embedding[:5] if embed_len > 0 else []
    logger.info(f"Querying ChromaDB: url={base_url}, collection={collection_id}")
    logger.info(f"Payload stats: embedding_dim={embed_len}, top_k={top_k}, filters={filters}")
    logger.info(f"Embedding sample: {embed_sample}")
    # logger.debug(f"Full payload: {json.dumps(payload)}") # Uncomment if needed, but might be huge
    
    if embed_len == 0:
        logger.error("Embedding vector is empty! Aborting query.")
        return {"ids": [], "distances": [], "metadatas": [], "documents": []}

    url = f"{base_url}/api/v1/collections/{collection_id}/query"
    response = requests.post(url, json=payload, timeout=30)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"ChromaDB Query Failed: {e}")
        logger.error(f"Response body: {response.text}")
        raise

    result = response.json()
    
    # Log summary of results
    ids = result.get("ids", [])
    count = len(ids[0]) if ids and len(ids) > 0 else 0
    logger.info(f"ChromaDB query returned {count} results.")
    return result

def _format_neighbors(chroma_response: Dict[str, Any], target_element: str, max_neighbors: int) -> List[Dict[str, Any]]:
    ids = chroma_response.get("ids", [[]])
    metadatas = chroma_response.get("metadatas", [[]])
    distances = chroma_response.get("distances", [[]])
    
    if not ids or not ids[0]:
        logger.info("No ids found in Chroma response.")
        return []

    neighbors = []
    normalized_element = target_element.upper()
    
    logger.info(f"Formatting neighbors for element={normalized_element}. Candidates: {len(ids[0])}")

    for idx, metadata, distance in zip(ids[0], metadatas[0], distances[0]):
        if not isinstance(metadata, dict):
            logger.debug(f"Skipping neighbor {idx}: metadata is not a dict.")
            continue
        
        meta_element = str(metadata.get("element", "")).upper()
        if meta_element != normalized_element:
            logger.debug(f"Skipping neighbor {idx}: element mismatch ({meta_element} != {normalized_element})")
            continue
        
        neighbor_payload = {
            "id": idx,
            "distance": distance,
            "metadata": metadata
        }
        neighbors.append(neighbor_payload)
        if len(neighbors) >= max_neighbors:
            break
            
    logger.info(f"Formatted {len(neighbors)} neighbors after filtering.")
    return neighbors

def _summarize_neighbors(neighbors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    abnormal_weight = 0.0
    total_weight = 0.0
    summarized = []

    for neighbor in neighbors:
        distance = float(neighbor.get("distance", 0))
        weight = 1.0 / (distance + 1e-6)
        total_weight += weight
        
        metadata = neighbor.get("metadata", {})
        classification = str(metadata.get("class", ""))
        if classification == "3": # Abnormal
            abnormal_weight += weight
            
        summarized.append({
            "station_id": metadata.get("region") or metadata.get("station_id"),
            "element": metadata.get("element"),
            "class_label": "이상" if classification == "3" else "정상" if classification == "0" else "미분류",
            "similarity": "유사" if distance < 2.0 else "다름",
            "original_start": metadata.get("original_start"),
            "original_end": metadata.get("original_end"),
        })

    abnormal_prob = (abnormal_weight / total_weight * 100.0) if total_weight > 0 else 0.0
    return {"abnormal_probability": abnormal_prob, "neighbors": summarized}

# =============================================================================
# Main Logic
# =============================================================================

def perform_abnormal_decision(
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
) -> Dict[str, Any]:
    settings = get_settings()
    
    # 1. Time Series Analysis
    element = element.upper()
    start_dt = _normalize_timestamp(start_time)
    end_dt = _normalize_timestamp(end_time)
    
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    query = {
        "type": "time_series",
        "region": int(station_id),
        "element": element,
        "start_time": start_str,
        "end_time": end_str,
    }
    
    # Fetch Original Data
    original_data = fetch_data(
        settings.postgres_user,
        settings.postgres_password,
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
        settings.double_the_sequence,
        settings.additional_days,
        query,
        settings.db_csv_path,
    )
    original_data["region"] = _to_native_scalar(original_data.get("region"))
    
    # Fetch Related Data
    related_results = []
    if INCLUDE_RELATED:
        related_ids = _get_station_network().get_related_station(int(station_id), element)
        if MAX_RELATED:
            related_ids = related_ids[:MAX_RELATED]
            
        for rid in related_ids:
            try:
                r_query = {**query, "region": int(rid)}
                r_data = fetch_data(
                    settings.postgres_user,
                    settings.postgres_password,
                    settings.postgres_host,
                    settings.postgres_port,
                    settings.postgres_db,
                    settings.double_the_sequence,
                    settings.additional_days,
                    r_query,
                    settings.db_csv_path,
                )
                r_data["region"] = _to_native_scalar(r_data.get("region"))
                related_results.append(r_data)
            except Exception:
                continue

    # Compute Similarity
    similarity_output = _compute_similarity_metrics(
        original_data,
        related_results,
        COMPARISON_TYPE,
        WINDOW_SIZE,
        MISSING_VALUE_POLICY,
    )
    
    comparisons = similarity_output.get("results", [])
    avg_confidence = 1.0
    confidences = [c.get("confidence") for c in comparisons if c.get("confidence") is not None]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)

    # 2. Insight Retrieval
    # Generate Embedding
    values = _parse_series_values(original_data)
    documents = [",".join("nan" if math.isnan(v) else str(v) for v in values)]
    
    weight_path = _resolve_weight_path(settings.trep_model_dir, element)
    device = _resolve_device_pool()[0]
    
    embedding_model = TRepEmbedding(
        weight_path=str(weight_path),
        device=device,
        encoding_window=os.getenv("NIER_TREP_ENCODING_WINDOW", "full_series"),
        time_embedding=os.getenv("NIER_TREP_TIME_EMBEDDING", "t2v_sin"),
    )
    
    embeddings = embedding_model(documents)
    if not embeddings:
        raise RuntimeError("Failed to generate embeddings")
    
    embedding_vector = _coerce_float_sequence(embeddings[0])
    
    # Vector Search
    host = settings.vector_db_host.rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    
    port = str(settings.vector_db_port)
    if port and f":{port}" not in host.split("//", 1)[-1]:
        base_url = f"{host}:{port}"
    else:
        base_url = host
    
    collection_name = settings.vector_collection or "trep"
    
    try:
        collection_id = _resolve_collection_id(base_url, collection_name)
        search_results = _query_chromadb(
            base_url,
            collection_id,
            embedding_vector,
            top_k=settings.vector_db_top_k,
            filters={"element": element}
        )
        neighbors = _format_neighbors(search_results, element, settings.vector_db_top_k)
        insight_summary = _summarize_neighbors(neighbors)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        insight_summary = {"abnormal_probability": 0.0, "neighbors": []}

    # 3. Response Orchestration
    abnormal_prob = insight_summary.get("abnormal_probability", 0.0)
    final_abnormal_prob = (abnormal_prob * 0.5) + ((1 - avg_confidence) * 50.0)
    final_abnormal_prob = max(0.0, min(100.0, final_abnormal_prob))
    
    # Format Output
    station_section = ""
    if comparisons:
        lines = []
        for idx, item in enumerate(comparisons, 1):
            sid = item.get("station_id", "N/A")
            lbl = item.get("z_score_range", "데이터 부족")
            comment = _Z_SCORE_COMMENTS.get(lbl, "비교 결과 부족")
            lines.append(f"- {idx}순위 연관 측정소 {sid} {comment}")
        station_section = "\n".join(lines)
    else:
        station_section = "연관 측정소가 없습니다."
        
    neighbor_section = ""
    neighbor_entries = insight_summary.get("neighbors", [])
    if neighbor_entries:
        lines = []
        for item in neighbor_entries:
            lines.append(
                f"- 기존 판정 결과 {item['station_id']} "
                f"(성분: {item['element']}, 상태: {item['class_label']}, "
                f"유사도: {item['similarity']}, 측정일시: {item['original_start']} ~ {item['original_end']})"
            )
        neighbor_section = "\n".join(lines)
    else:
        neighbor_section = "유사 기존 판정 결과가 없습니다."

    station_section += "\n### 종합 판정 확률 (연관 측정소 및 기존 판정 결과 반영) ###\n"
    station_section += f"\n- 베이스라인 이상 확률: {final_abnormal_prob:.2f}%\n"

    prompt = (
        "## 데이터 판정 요청 ##\n"
        f"다음은 측정소 {station_id} 에서 {start_str} 부터 {end_str} 까지 수집된 {element} 성분의 비교 결과입니다.\n\n"
        "### 유사한 기존 판정 결과 ###\n"
        f"{neighbor_section}\n\n"
        "### 연관 측정소 비교 결과 ###\n"
        f"{station_section}\n"
        "### 요청 사항 ###\n"
        "위 데이터를 종합적으로 분석하여, 해당 데이터가 정상인지 베이스라인 이상인지 판정하고 이유를 설명해주세요.\n"
        "연관 측정소 비교 결과와 유사한 기존 판정 결과의 지역번호, 성분, 상태, 유사도, 측정일시를 사용자에게 보여주세요\n"
    )

    return {
        "answer": prompt,
        "graph_image": {
            "type": "image",
            "url": "https://github.com/user-attachments/assets/9e7ba293-8c0e-4e98-a81c-4290a2dd1300",
            "mimeType": "image/png",
        }
    }
