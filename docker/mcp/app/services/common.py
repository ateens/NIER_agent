import math
from typing import Any, Dict, List, Optional, Sequence


def ensure_sequence(values: Optional[Sequence[Any]]) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    return list(values)


def parse_series_values(series: Optional[Dict[str, Any]]) -> List[float]:
    """Convert a values payload (comma-separated string or iterable) into floats."""
    if not series:
        return []

    raw_values = series.get("values")
    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        tokens = [token.strip() for token in raw_values.split(",")]
    else:
        tokens = [token for token in ensure_sequence(raw_values)]

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


def coerce_value_payload(values: Optional[Sequence[Any]]) -> List[float]:
    """Normalize arbitrary value payloads into a float list for embedding generation."""
    if values is None:
        return []
    normalized: List[float] = []
    for item in values:
        if isinstance(item, dict):
            normalized.extend(parse_series_values(item))
        else:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                continue
    return normalized
