"""Generation-only literature search with saved query receipts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

from mcp.server.fastmcp import FastMCP
import requests


ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,url,openAccessPdf,externalIds"
_lock = threading.Lock()
_last_request = 0.0
mcp = FastMCP("research", instructions="Search physical-asset literature; query receipts are saved automatically.")


@mcp.tool()
def search_papers(query: str, limit: int = 5) -> dict:
    """Search Semantic Scholar for papers, abstracts and open-access links.

    Save the query and result under data/research. Read relevant primary sources
    before using a result to justify a failure mapping or diagnostic method.
    """
    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("Provide a nonempty query and a limit from 1 to 100")
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    record = {"provider": "semantic_scholar", "endpoint": ENDPOINT,
              "query": query, "limit": limit, "authenticated": bool(key),
              "started_at": datetime.now(timezone.utc).isoformat(), "attempts": []}
    global _last_request
    with _lock:
        for attempt in range(3):
            time.sleep(max(0, 1.05 - (time.monotonic() - _last_request)))
            try:
                response = requests.get(ENDPOINT, params={"query": query, "limit": limit, "fields": FIELDS},
                                        headers=headers, timeout=30, allow_redirects=False)
                record["attempts"].append(response.status_code)
                if response.status_code == 200:
                    result = response.json()
                    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                        raise ValueError("Malformed search result")
                    record.pop("error", None)
                    record.update(status="ok", result=result)
                    break
                record.update(status="error", error=f"Semantic Scholar HTTP {response.status_code}")
                if response.status_code != 429:
                    break
            except (requests.RequestException, ValueError):
                record.update(status="error", error="Semantic Scholar request failed or returned invalid JSON")
                break
            finally:
                _last_request = time.monotonic()
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    path = Path("data/research") / f"search-{uuid.uuid4().hex[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record["response_file"] = str(path)
    path.write_text(json.dumps(record, indent=2) + "\n")
    receipt = {k: v for k, v in record.items() if k != "result"}
    receipt["paper_count"] = len(record.get("result", {}).get("data", []))
    log = Path(os.environ.get("SCENARIO_RESEARCH_LOG", "data/research/queries.jsonl"))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as output:
        output.write(json.dumps(receipt) + "\n")
    return record


if __name__ == "__main__":
    mcp.run()
