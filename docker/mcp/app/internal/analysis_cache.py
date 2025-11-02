from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional
from uuid import uuid4


class _AnalysisCache:
    """In-memory LRU cache for time-series payloads shared between tools."""

    def __init__(self, capacity: int = 256):
        self._capacity = max(1, capacity)
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def store(self, payload: Dict[str, Any]) -> str:
        key = uuid4().hex
        with self._lock:
            self._store[key] = payload
            self._store.move_to_end(key)
            self._evict_if_needed()
        return key

    def get(self, key: str) -> Dict[str, Any]:
        with self._lock:
            if key not in self._store:
                raise KeyError(key)
            payload = self._store[key]
            self._store.move_to_end(key)
            return payload

    def pop(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.pop(key, None)

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)


_CACHE = _AnalysisCache()


def cache_series_payload(payload: Dict[str, Any]) -> str:
    """Store a time-series payload and return a reusable cache key."""

    return _CACHE.store(payload)


def load_series_payload(cache_key: str) -> Dict[str, Any]:
    """Retrieve a cached payload by key."""

    return _CACHE.get(cache_key)


def pop_series_payload(cache_key: str) -> Optional[Dict[str, Any]]:
    """Remove a cached payload when it is no longer needed."""

    return _CACHE.pop(cache_key)

