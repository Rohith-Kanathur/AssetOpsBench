"""Snapshot or restore CouchDB inside a private benchmark container."""

import argparse
import json
import os
from pathlib import Path

import requests


def transfer(action, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.auth = (os.environ["COUCHDB_USERNAME"], os.environ["COUCHDB_PASSWORD"])
    base = os.environ["COUCHDB_URL"].rstrip("/")

    def request(method, path, **kwargs):
        response = session.request(method, base + path, timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()

    if action == "export":
        names = [n for n in request("GET", "/_all_dbs") if not n.startswith("_")]
        for name in names:
            rows = request("GET", f"/{name}/_all_docs", params={"include_docs": "true", "attachments": "true"})["rows"]
            docs = [{k: v for k, v in row["doc"].items() if k != "_rev"}
                    for row in rows if not row.get("value", {}).get("deleted")]
            (directory / f"{name}.json").write_text(json.dumps(docs) + "\n")
    else:
        for path in sorted(directory.glob("*.json")):
            request("PUT", "/" + path.stem)
            docs = json.loads(path.read_text())
            for start in range(0, len(docs), 500):
                results = request("POST", f"/{path.stem}/_bulk_docs", json={"docs": docs[start:start + 500]})
                if any("error" in row for row in results):
                    raise ValueError(f"Failed to restore {path.stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "restore"))
    parser.add_argument("directory")
    args = parser.parse_args()
    transfer(args.action, args.directory)
