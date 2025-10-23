from contextlib import contextmanager
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from sqlalchemy import create_engine

from internal.NIERModules.AIS import (
    AisDataAir,
    AisDataAbnrm,
    ReadOnlySession,
    dnsty_of,
    MATTER_FLAG_MAP,
    CNTMN_CODE_MAP,
)
from internal.NIERModules.query_parser import validate_query

import pandas as pd
import os


def load_pkl_for_range(db_path: str, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Load PKL files for the given range of years.
    """
    start_year = datetime.strptime(start_time.split(" ")[0], "%Y-%m-%d").year
    end_year = datetime.strptime(end_time.split(" ")[0], "%Y-%m-%d").year
    years = set(range(start_year, end_year + 1))

    data_list = []
    for year in years:
        pkl_path = os.path.join(db_path, f"{year}.pkl")
        if os.path.exists(pkl_path):
            print("###LOAD_PKL### Loading PKL file:", pkl_path)
            df = pd.read_pickle(pkl_path)
            data_list.append(df)
        else:
            print(f"No PKL found for {year}, skipping...")

    if data_list:
        return pd.concat(data_list, ignore_index=True)
    return pd.DataFrame()


@contextmanager
def get_readonly_db(
    POSTGRESQL_USER: str,
    POSTGRESQL_PASSWORD: str,
    POSTGRESQL_URL: str,
    POSTGRESQL_PORT: str,
    POSTGRESQL_DB: str,
):
    """Create a read-only database session."""
    POSTGRESQL_DATABASE_URL = (
        f"postgresql://{quote_plus(POSTGRESQL_USER)}:"
        f"{quote_plus(POSTGRESQL_PASSWORD)}@"
        f"{quote_plus(POSTGRESQL_URL)}:"
        f"{quote_plus(POSTGRESQL_PORT)}/"
        f"{quote_plus(POSTGRESQL_DB)}"
    )

    engine = create_engine(POSTGRESQL_DATABASE_URL, echo=False)
    ReadOnlySessionLocal = ReadOnlySession(autocommit=False, autoflush=False, bind=engine)

    session = ReadOnlySessionLocal()
    try:
        yield session
    finally:
        session.close()


def fetch_data(
    POSTGRESQL_USER: str,
    POSTGRESQL_PASSWORD: str,
    POSTGRESQL_URL: str,
    POSTGRESQL_PORT: str,
    POSTGRESQL_DB: str,
    DOUBLE_THE_SEQUENCE: bool,
    ADDITIONAL_DAYS: int,
    query: dict,
    db_path: str = "",
) -> dict:
    """
    Fetches time-series data based on the query parameters.
    """
    validate_query(query)
    start_time, end_time, element, station = (
        query["start_time"],
        query["end_time"],
        query["element"],
        query["region"],
    )
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    if DOUBLE_THE_SEQUENCE:
        duration = end_dt - start_dt
        adjusted_start_dt = start_dt - duration
    else:
        adjusted_start_dt = start_dt - timedelta(days=ADDITIONAL_DAYS)

    adjusted_start_time = adjusted_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    start_time = adjusted_start_time

    print("###FETCH_DATA###", flush=True)
    if db_path:
        try:
            df = pd.read_csv(db_path)
            filtered_df = df[
                (df["AREA_INDEX"] == station)
                & (df["MDATETIME"] >= start_time)
                & (df["MDATETIME"] <= end_time)
            ]
            values = filtered_df[element].tolist()
            print("CSV loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"CSV Data Fetch Error: {e}")
    else:
        print("###FETCH_DATA### Fetching data from PostgreSQL...")
        with get_readonly_db(
            POSTGRESQL_USER,
            POSTGRESQL_PASSWORD,
            POSTGRESQL_URL,
            POSTGRESQL_PORT,
            POSTGRESQL_DB,
        ) as db:
            element_column = dnsty_of(element, AisDataAir).label("VALUE")
            flag_column = getattr(AisDataAir, MATTER_FLAG_MAP[element]).label(element + "_FLAG")

            stmt = (
                db.query(
                    AisDataAir.data_knd_cd.label("DATA_CD"),
                    AisDataAir.msrmt_ymdh.label("MDATETIME"),
                    AisDataAir.msrstn_cd.label("AREA_INDEX"),
                    element_column,
                    flag_column,
                )
                .filter(AisDataAir.data_knd_cd == "DATAR1")
                .filter(AisDataAir.msrstn_cd == station)
                .filter(
                    AisDataAir.msrmt_ymdh
                    >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H")
                )
                .filter(
                    AisDataAir.msrmt_ymdh
                    <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H")
                )
            )

            print("###DEBUG### SQL Query Parameters:", flush=True)
            print(f"  - station: {station} (type: {type(station)})", flush=True)
            print(f"  - start_time: {start_time} (type: {type(start_time)})", flush=True)
            print(f"  - end_time: {end_time} (type: {type(end_time)})", flush=True)
            print(f"  - element: {element}", flush=True)
            print("###DEBUG### Generated SQL:", flush=True)
            print(str(stmt), flush=True)

            df = pd.read_sql_query(stmt.statement, db.bind)

            print(f"###DEBUG### 1. Raw DataFrame shape: {df.shape}", flush=True)
            print(f"###DEBUG### 1. Raw DataFrame columns: {df.columns.tolist()}", flush=True)
            if len(df) > 0:
                print("###DEBUG### 1. First 3 rows:", flush=True)
                print(df.head(3), flush=True)
                print(f"###DEBUG### 1. VALUE column type: {df['VALUE'].dtype}", flush=True)
                print(f"###DEBUG### 1. VALUE non-null count: {df['VALUE'].count()}", flush=True)
                print(f"###DEBUG### 1. VALUE sample values: {df['VALUE'].head().tolist()}", flush=True)
            else:
                print("###DEBUG### 1. DataFrame is empty!", flush=True)

            df["MDATETIME"] = pd.to_datetime(df["MDATETIME"], format="%Y%m%d%H", errors="coerce")

            print(f"###DEBUG### 2. Flag column: {element}_FLAG", flush=True)
            if len(df) > 0 and f"{element}_FLAG" in df.columns:
                print("###DEBUG### 2. Flag distribution:", flush=True)
                print(df[f"{element}_FLAG"].value_counts(), flush=True)

            df.loc[df[f"{element}_FLAG"] != 1, "VALUE"] = float("nan")

            print(
                f"###DEBUG### 3. After flag filtering - VALUE non-null count: {df['VALUE'].count()}",
                flush=True,
            )

            wrong_stmt = (
                db.query(
                    AisDataAbnrm.msrmt_ymdh.label("MDATETIME"),
                    AisDataAbnrm.msrstn_cd.label("AREA_INDEX"),
                    AisDataAbnrm.cntmn_dtl_cd.label("ELEMENT_CD"),
                    AisDataAbnrm.abnrm_data_se_cd.label("WRONG_CODE"),
                )
                .filter(AisDataAbnrm.msrmt_ymdh >= start_time)
                .filter(AisDataAbnrm.msrmt_ymdh <= end_time)
            )
            wrong_df = pd.read_sql_query(wrong_stmt.statement, db.bind)

            wrong_df["MDATETIME"] = pd.to_datetime(
                wrong_df["MDATETIME"],
                format="%Y%m%d%H",
                errors="coerce",
            )
            wrong_df["ELEMENT"] = wrong_df["ELEMENT_CD"].map(CNTMN_CODE_MAP)

            df = df.merge(
                wrong_df[["MDATETIME", "AREA_INDEX", "ELEMENT", "WRONG_CODE"]],
                on=["MDATETIME", "AREA_INDEX"],
                how="left",
            )
            df[f"{element}_LABEL"] = df["WRONG_CODE"].fillna(0).astype(int)
            df.drop(columns=["WRONG_CODE"], inplace=True)

            print("df['VALUE'].head()", df["VALUE"].head(), flush=True)
            values = df["VALUE"].tolist()
            print(f"###DEBUG### 4. Final values count: {len(values)}", flush=True)
            print(
                f"###DEBUG### 4. Final values sample: {values[:10] if values else 'EMPTY'}",
                flush=True,
            )

    values = [float("nan") if v == 999999.0 else v for v in values]
    print(f"###DEBUG### 5. After 999999 filtering: {len(values)} values", flush=True)
    return {
        "region": station,
        "start_time": start_time,
        "end_time": end_time,
        "element": element,
        "values": ",".join(map(str, values)),
    }
