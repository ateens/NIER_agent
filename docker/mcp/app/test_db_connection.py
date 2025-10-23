"""
Quick connection check for the MCP PostgreSQL or CSV data source.

Usage (inside the `mcp` container):
    python test_db_connection.py [station_id element start_date end_date]

If no arguments are provided the script will use a short default range.
"""

from __future__ import annotations

import sys
from pprint import pprint

try:
    from config import get_settings
    from services.timeseries_service import perform_timeseries_analysis
except ImportError as exc:  # pragma: no cover - convenience script
    print(f"Import error: {exc}")
    sys.exit(1)


def main(argv: list[str]) -> None:
    settings = get_settings()
    station_id = int(argv[1]) if len(argv) > 1 else 111261
    element = argv[2] if len(argv) > 2 else "SO2"
    start_time = argv[3] if len(argv) > 3 else "2024-03-01"
    end_time = argv[4] if len(argv) > 4 else "2024-03-07"

    print("=== Environment summary ===")
    pprint(
        {
            "POSTGRESQL_HOST": settings.postgres_host,
            "POSTGRESQL_PORT": settings.postgres_port,
            "POSTGRESQL_DB": settings.postgres_db,
            "POSTGRESQL_USER": settings.postgres_user,
            "TREP_MODEL_DIR": settings.trep_model_dir,
            "VECTOR_DB_HOST": settings.vector_db_host,
        }
    )

    print(
        f"\nAttempting timeseries_analysis "
        f"(station={station_id}, element={element}, "
        f"{start_time}~{end_time})..."
    )

    try:
        result = perform_timeseries_analysis(
            settings,
            station_id=station_id,
            element=element,
            start_time=start_time,
            end_time=end_time,
            include_related=False,
            compute_similarity=False,
        )
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"\n❌ Query failed: {type(exc).__name__}: {exc}")
        raise

    original = result.get("original", {})
    values = original.get("values", "")
    sample = values.split(",")
    print("\n✅ Query succeeded.")
    print(f"Original series length: {len(values.split(','))}")
    print(f"Sample values: {sample}")

    if result.get("related_errors"):
        print("\nRelated station errors:")
        for item in result["related_errors"]:
            print(f" - {item}")


if __name__ == "__main__":
    main(sys.argv)
