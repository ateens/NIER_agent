from typing import Any, Dict, List, Optional, Sequence

from config import get_settings
from .common import coerce_value_payload, ensure_sequence

__all__ = ["build_insight_payload"]


def _ensure_embedding_source(
    values: Optional[Sequence[Any]],
    element: Optional[str],
) -> List[float]:
    floats = coerce_value_payload(values)
    if floats:
        return floats

    # Fall back to generic placeholder when no values provided
    element_tag = (element or "GENERIC").upper()
    return [0.0, 0.0, hash(element_tag) % 10 / 10.0]


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
    settings = get_settings()

    metadata: Dict[str, Any] = {
        "perform_embedding": perform_embedding,
        "perform_search": perform_search,
        "collection": collection or settings.vector_collection,
        "vector_db_host": settings.vector_db_host,
        "vector_db_port": settings.vector_db_port,
        "top_k": top_k,
        "filters": filters or {},
        "element": element,
    }

    generated_embedding: Optional[List[float]] = embedding

    if perform_embedding:
        generated_embedding = _ensure_embedding_source(values, element)
        metadata.update(
            {
                "embedding_source": "generated_stub",
                "input_value_count": len(ensure_sequence(values)),
                "embedding_dim": len(generated_embedding),
            }
        )
    else:
        if generated_embedding is None:
            raise ValueError("embedding must be provided when perform_embedding=False")
        metadata.update(
            {
                "embedding_source": "provided",
                "input_value_count": len(generated_embedding),
                "embedding_dim": len(generated_embedding),
            }
        )

    neighbors: List[Dict[str, Any]] = []
    if perform_search:
        neighbors = [
            {
                "id": f"placeholder-{idx}",
                "distance": float(idx) * 0.1,
                "metadata": {"element": element, "stub": True},
                "document": None,
                "embedding": generated_embedding,
            }
            for idx in range(min(top_k, 3))
        ]
        metadata["returned_neighbors"] = len(neighbors)

    return {
        "embedding": generated_embedding,
        "neighbors": neighbors,
        "metadata": metadata,
    }
