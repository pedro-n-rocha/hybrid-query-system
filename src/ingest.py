import json
import sqlite3
import pandas as pd
from flatten_json import flatten


def load(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest(data: dict, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    vulns = data.get("vulnerabilities", [])
    #    vulns = vulns[:10]  # limit to 10 for now , expand later to the full

    flat = [flatten(v, separator="_") for v in vulns]  # flatten nested stuff

    df = pd.DataFrame(flat)

    keep = {
        "cve_id": "cve_id",
        "cve_descriptions_0_value": "description",
        "cve_metrics_cvssMetricV31_0_cvssData_baseScore": "cvss_v3",
        "cve_metrics_cvssMetricV31_0_cvssData_baseSeverity": "severity",
        "cve_weaknesses_0_description_0_value": "cwe",
        "cve_published": "published",
        "cve_lastModified": "last_modified",
    }

    cols = [k for k in keep if k in df.columns]
    df = df[cols].rename(columns=keep)

    df.to_sql("cves", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()


def main():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/cves/nvdcve-2.0-recent.json")
    ap.add_argument("--db", default="data/db/docstore.db")
    args = ap.parse_args()

    d = load(args.json)
    ingest(d, args.db)


if __name__ == "__main__":
    main()
