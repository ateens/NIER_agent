import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
    vector_db_host: str
    vector_db_port: str
    vector_collection: str
    vector_db_top_k: int
    trep_model_dir: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load reusable environment-driven settings."""
    pg_host = os.getenv("POSTGRESQL_HOST", "localhost")
    max_related = os.getenv("NIER_MAX_RELATED_STATIONS")

    base_dir = Path(__file__).resolve()
    default_trep_dir = base_dir.parent / "internal" / "NIERModules" / "chroma_trep" / "model_pkl"
    trep_model_dir = Path(os.getenv("TREP_MODEL_DIR", str(default_trep_dir))).expanduser()

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
        vector_db_host=os.getenv("VECTOR_DB_HOST", "http://localhost"),
        vector_db_port=os.getenv("VECTOR_DB_PORT", "8000"),
        vector_collection=os.getenv("VECTOR_COLLECTION_NAME", "time_series_collection_trep"),
        vector_db_top_k=int(os.getenv("VECTOR_DB_TOP_K", "10")),
        trep_model_dir=str(trep_model_dir),
    )
