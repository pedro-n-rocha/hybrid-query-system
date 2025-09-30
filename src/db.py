import sqlite3
import pandas as pd


def fetch_cves_by_ids(db_path: str, ids: list[str]) -> pd.DataFrame:

    placeholders = ",".join(["?"] * len(ids))
    query = f"""
        SELECT *
        FROM cves
        WHERE cve_id IN ({placeholders})
    """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=ids)
    conn.close()

    df["__order"] = df["cve_id"].apply(lambda x: ids.index(x))
    df = df.sort_values("__order").drop(columns="__order")
    return df
