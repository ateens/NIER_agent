import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: str
    postgres_db: str
    double_the_sequence: bool
    additional_days: int
    db_csv_path: str
    max_related: Optional[int]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load reusable environment-driven settings."""
    pg_host = (
        os.getenv("POSTGRESQL_URL")
        or os.getenv("POSTGRESQL_HOST")
        or "localhost"
    )
    max_related = os.getenv("NIER_MAX_RELATED_STATIONS")
    return Settings(
        postgres_user=os.getenv("POSTGRESQL_USER", "inha"),
        postgres_password=os.getenv("POSTGRESQL_PASSWORD", "inha3345!!"),
        postgres_host=pg_host,
        postgres_port=os.getenv("POSTGRESQL_PORT", "5432"),
        postgres_db=os.getenv("POSTGRESQL_DB", "airinfo"),
        double_the_sequence=_str_to_bool(os.getenv("DOUBLE_THE_SEQUENCE"), False),
        additional_days=int(os.getenv("ADDITIONAL_DAYS", "14")),
        db_csv_path=os.getenv("NIER_TIMESERIES_CSV_PATH", ""),
        max_related=int(max_related) if max_related is not None else None,
    )
