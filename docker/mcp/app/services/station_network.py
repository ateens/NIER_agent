import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class StationNetwork:
    """Lightweight loader for station similarity groups."""

    def __init__(self, similarity_path: Optional[str] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        default_path = base_dir.parent / "vendor" / "assets" / "similarity_results_v2.pkl"

        legacy_path = None
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
            raise FileNotFoundError(
                "Could not load station similarity data. "
                "Place 'similarity_results_v2.pkl' under docker/mcp/app/vendor/assets "
                "or set NIER_STATION_SIMILARITY_PATH."
            )

    @staticmethod
    def _load_pickle(path: Path) -> Dict[int, Dict[str, Dict[int, Dict[str, float]]]]:
        with path.open("rb") as handle:
            data = pickle.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected station similarity payload type: {type(data)}")
        return data

    def get_station_list(self) -> List[int]:
        return list(self._groups.keys())

    def station_exists(self, station_id: int) -> bool:
        return int(station_id) in self._groups

    def get_related_station(self, station_id: int, element: str) -> List[int]:
        station_id = int(station_id)
        element = element.upper()

        if not self.station_exists(station_id):
            return []
        element_map = self._groups.get(station_id, {})
        if element not in element_map:
            return []

        return list(element_map[element].keys())

    def get_similarity_stats(
        self,
        station_a: int,
        station_b: int,
        element: str,
        window_size: int,
    ) -> Optional[Dict[str, float]]:
        """Return baseline FastDTW statistics for a station pair."""
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
