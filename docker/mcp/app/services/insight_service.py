from typing import Any, Dict, List, Optional, Sequence

from .common import coerce_value_payload

__all__ = ["build_insight_payload"]


def build_insight_payload(
    values: Optional[Sequence[Any]] = None,
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
    metadata: Dict[str, Any] = {
        "element": element,
        "collection": collection,
        "perform_embedding": perform_embedding,
        "perform_search": perform_search,
        "device": device or "cuda:0",
        "top_k": top_k,
    }

    generated_embedding: Optional[List[float]] = embedding
    if perform_embedding:
        parsed_values = coerce_value_payload(values)
        metadata["embedding_source"] = "generated" if parsed_values else "empty_input"
        generated_embedding = []  # TODO: integrate actual T-Rep inference
        metadata["input_value_count"] = len(parsed_values)
    else:
        metadata["embedding_source"] = "provided"
        metadata["input_value_count"] = len(generated_embedding or [])

    neighbors: List[Dict[str, Any]] = []
    if perform_search:
        metadata["search_filters"] = filters or {}
        neighbors = []  # TODO: integrate Chroma/Vector DB search

    return {
        "embedding": generated_embedding,
        "neighbors": neighbors,
        "metadata": metadata,
    }
