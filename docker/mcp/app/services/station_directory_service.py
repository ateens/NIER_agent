import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import ensure_sequence

ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "station_directory.json"


@lru_cache(maxsize=1)
def _load_station_records() -> List[Dict[str, Any]]:
    with ASSET_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    normalized = token.strip()
    return normalized if normalized else None


def fetch_station_directory(
    *,
    sido: Optional[str] = None,
) -> Dict[str, Any]:
    records = _load_station_records()

    target_sido = _normalize_token(sido)
    
    # "전체" 옵션 처리: sido가 "전체"인 경우 None으로 처리
    if target_sido == "전체":
        target_sido = None

    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}

    for entry in records:
        entry_sido = str(entry.get("시/도", "")).strip()
        entry_city = str(entry.get("도시", "")).strip()
        station_code = str(entry.get("측정소코드", "")).strip()
        station_name = str(entry.get("측정소명", "")).strip()

        if target_sido and entry_sido != target_sido:
            continue

        city_bucket = grouped.setdefault(entry_sido, {})
        station_map = city_bucket.setdefault(entry_city, {})
        station_map.setdefault(station_code, station_name)

    regions: List[Dict[str, Any]] = []
    total_stations = 0
    total_cities = 0

    for region_name in sorted(grouped.keys()):
        cities_payload: List[Dict[str, Any]] = []
        for city_name in sorted(grouped[region_name].keys()):
            station_map = grouped[region_name][city_name]
            stations = [
                {"code": code, "name": station_map[code]}
                for code in sorted(station_map.keys())
            ]
            total_stations += len(stations)
            total_cities += 1
            cities_payload.append(
                {
                    "city": city_name,
                    "stations": stations,
                }
            )

        regions.append(
            {
                "sido": region_name,
                "cities": cities_payload,
            }
        )

    metadata = {
        "filters": {
            "sido": target_sido,
        },
        "total_regions": len(regions),
        "total_cities": total_cities,
        "total_stations": total_stations,
        "source_path": str(ASSET_PATH),
    }

    return {
        "regions": regions,
        "metadata": metadata,
    }
