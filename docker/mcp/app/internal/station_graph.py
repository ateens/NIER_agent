"""Utility helpers for visualizing station comparisons."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from config import Settings
from services.common import parse_series_values
from services.timeseries_service import select_related_stations
from vendor.modules.NIER.postgres_handler import fetch_data


def _build_time_axis(start: str, end: str, count: int) -> List[datetime]:
    if count <= 0:
        return []
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    if count == 1 or start_dt == end_dt:
        return [start_dt]
    span = end_dt - start_dt
    step = span / max(count - 1, 1)
    return [start_dt + step * idx for idx in range(count)]


def _fetch_series(
    settings: Settings,
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
) -> Tuple[List[float], Dict[str, str]]:
    query = {
        "type": "time_series",
        "region": int(station_id),
        "element": element,
        "start_time": start_time,
        "end_time": end_time,
    }
    payload = fetch_data(
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
    values = parse_series_values({"values": payload.get("values", "")})
    return values, payload


def plot_station_comparison(
    station_id: int,
    element: str,
    start_time: str,
    end_time: str,
    *,
    related_limit: int = 10,
    figsize: Tuple[float, float] = (12, 6),
    settings: Optional[Settings] = None,
) -> plt.Figure:
    """Plot the target station and its related stations for quick comparison.

    Parameters
    ----------
    station_id : int
        기준 측정소 ID.
    element : str
        관측 성분(예: SO2, PM10 등).
    start_time, end_time : str
        조회 구간. `YYYY-MM-DD HH:MM:SS` 형식을 권장합니다.
    related_limit : int, optional
        비교에 사용할 연관 측정소 최대 개수.
    figsize : tuple(float, float), optional
        Matplotlib figure 크기.
    settings : Settings, optional
        명시하지 않으면 기본 Settings()를 사용합니다.

    Returns
    -------
    matplotlib.figure.Figure
        생성된 그래프 Figure. 필요하면 `savefig`/`show`로 후처리하세요.
    """

    cfg = settings or Settings()
    element_upper = element.upper()

    base_values, base_payload = _fetch_series(
        cfg,
        station_id=station_id,
        element=element_upper,
        start_time=start_time,
        end_time=end_time,
    )
    if not base_values:
        raise ValueError("기준 측정소 시계열이 비어 있습니다.")

    timeline = _build_time_axis(start_time, end_time, len(base_values))

    related_ids = select_related_stations(
        station_id=int(station_id),
        element=element_upper,
        max_related=related_limit,
    )

    related_series: List[Tuple[int, List[float]]] = []
    for related_id in related_ids:
        try:
            series_values, _ = _fetch_series(
                cfg,
                station_id=related_id,
                element=element_upper,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as exc:  # pragma: no cover - diagnostic utility
            print(f"[station_graph] 연관 측정소 {related_id} 조회 실패: {exc}")
            continue
        if not series_values:
            continue
        related_series.append((related_id, series_values))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        timeline,
        base_values,
        label=f"{station_id} (base)",
        linewidth=2.2,
        color="#1f77b4",
    )

    for related_id, values in related_series:
        aligned_values = values[: len(timeline)]
        ax.plot(
            timeline[: len(aligned_values)],
            aligned_values,
            label=f"{related_id} (related)",
            alpha=0.45,
            linewidth=1.2,
            linestyle="-",
        )

    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    return fig

if __name__ == "__main__":
    plot_station_comparison(
        station_id=22121,
        element="SO2",
        start_time="2024-01-01 00:00:00",
        end_time="2024-01-01 23:59:59",
    )