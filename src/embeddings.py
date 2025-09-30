from fastembed import TextEmbedding
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite_vec


def load_db(db_path: str) -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT * FROM cves ORDER BY cve_id ASC
        """,
        conn,
    )
    conn.close()
    return df


def load_model() -> TextEmbedding:
    m = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="models/embeddings",
    )
    return m


def gen_embeddings(model: TextEmbedding, df: pd.DataFrame):
    cve_ids = df["cve_id"].tolist()
    descriptions = df["description"].tolist()

    emb_gen = model.embed(descriptions, batch_size=32)
    vectors = [
        np.asarray(v, dtype=np.float32)
        for v in tqdm(
            model.embed(descriptions, batch_size=32),
            total=len(descriptions),
            desc="Embedding CVEs",
        )
    ]
    return pd.DataFrame({"cve_id": cve_ids, "vector": vectors})


def save_vec_db(df, db_path: str, dim: int = 384):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    dim = 384  # equal to embedding model

    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS cve_vectors
        USING vec0(
          cve_id TEXT PRIMARY KEY,
          embedding FLOAT[{dim}] distance_metric=cosine
        )
    """
    )

    rows = [(row.cve_id, row.vector) for row in df.itertuples(index=False)]
    conn.executemany(
        f"INSERT OR REPLACE INTO cve_vectors(cve_id, embedding) VALUES (?, ?)", rows
    )
    conn.commit()
    conn.close()


def vec_search(model: TextEmbedding, db_path: str, query: str):

    q_vec = np.asarray(list(model.embed([query]))[0], dtype=np.float32)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    q_blob = q_vec.tobytes()

    rows = conn.execute(
        """
    SELECT cve_id, distance
    FROM cve_vectors
    WHERE embedding MATCH vec_f32(?)
      AND k = 5
    """,
        (sqlite3.Binary(q_blob),),  # single param, note the comma!
    ).fetchall()

    conn.close()

    results = [(cid, float(1.0 - dist)) for (cid, dist) in rows]
    return results


def main():

    import argparse
    from fastembed import TextEmbedding

    ap = argparse.ArgumentParser()

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--db", default="data/db/docstore.db")
    p_build.add_argument("--model", default="models/tfidf")

    p_search = sub.add_parser("search")
    p_search.add_argument("--db", default="data/db/docstore.db")
    p_search.add_argument("--query", required=True)

    args = ap.parse_args()

    if args.cmd == "build":

        m = load_model()
        data = load_db(args.db)
        vec = gen_embeddings(m, data)
        save_vec_db(vec, args.db)

    elif args.cmd == "search":

        m = load_model()
        res = vec_search(m, args.db, args.query)

        for score_id, (cve_id, score) in enumerate(res, start=1):
            print(f"{score:0.6f}  {cve_id}")


if __name__ == "__main__":
    main()
