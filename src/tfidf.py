import sqlite3
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


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


def tfidf_fit(df, models_dir: str) -> None:
    cve_ids = df["cve_id"].tolist()
    docs = df["description"].tolist()

    vec = TfidfVectorizer(
        # max_features=None,  # heavy but compute for all
        min_df=1,  # keep rare tokens (default)
        max_df=0.95,  # drop super-common across docs
        ngram_range=(1, 2),
        lowercase=True,
        norm="l2",
        dtype=np.float32,
    )
    X = vec.fit_transform(docs)

    bundle = {"vectorizer": vec, "matrix": X, "ids": cve_ids}
    joblib.dump(bundle, f"{models_dir}/tfidf_index.joblib")  # save idx


def tfidf_load(models_dir: str):
    bundle = joblib.load(f"{models_dir}/tfidf_index.joblib")
    return bundle["vectorizer"], bundle["matrix"], bundle["ids"]


def tfidf_search(
    vec: TfidfVectorizer, X: sparse.csr_matrix, ids: list[str], query: str
):
    q = vec.transform([query])
    scores = (X @ q.T).toarray().ravel()
    idx = np.argsort(-scores)[:5]  # top 5 only
    return [(ids[i], float(scores[i])) for i in idx]


def main():
    import argparse

    ap = argparse.ArgumentParser()

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--db", default="data/db/docstore.db")
    p_build.add_argument("--model", default="models/tfidf")

    p_search = sub.add_parser("search")
    p_search.add_argument("--model", default="models/tfidf")
    p_search.add_argument("--query", required=True)

    args = ap.parse_args()

    if args.cmd == "build":
        df = load_db(args.db)
        tfidf_fit(df, args.model)

    elif args.cmd == "search":
        vec, X, ids = tfidf_load(args.model)
        results = tfidf_search(vec, X, ids, args.query)
        for score_id, (cve_id, score) in enumerate(results, start=1):
            print(f"{score:0.6f}  {cve_id}")


if __name__ == "__main__":
    main()
