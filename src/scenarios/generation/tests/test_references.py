"""Keep the small example bundle traceable to its existing corpus rows."""

import hashlib
import json
from pathlib import Path
import re

from scenarios.generation.budget import DOMAINS


PACKAGE = Path(__file__).resolve().parents[1]
EXAMPLES = json.loads((PACKAGE / "references/examples.json").read_text())["examples"]
SOURCE_ROWS = json.loads((PACKAGE / "references/source-rows.json").read_text())
HOLDOUT = json.loads((Path(__file__).parent / "fixtures/chiller-human-holdout.json").read_text())


def test_examples_retain_published_source_row_provenance():
    identities = {(example["source"]["path"], example["source"]["row_id"]) for example in EXAMPLES}
    assert len(identities) == len(EXAMPLES)
    assert len({path for path, _ in identities}) >= 5
    for example in EXAMPLES:
        assert example["type"] in DOMAINS
        source = example["source"]
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", source["file_sha256"])
        assert source["url"] == f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{source['path']}"
        row = next(item["row"] for item in SOURCE_ROWS
                   if item["path"] == source["path"] and item["row"]["id"] == source["row_id"])
        serialized = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert hashlib.sha256(serialized).hexdigest() == source["sha256"]


def test_examples_preserve_original_questions_and_characteristic_forms_verbatim():
    texts = [example["text"] for example in EXAMPLES]
    assert len(set(texts)) == len(texts)
    for example in EXAMPLES:
        source = example["source"]
        row = next(item["row"] for item in SOURCE_ROWS
                   if item["path"] == source["path"] and item["row"]["id"] == source["row_id"])
        for field in ("text", "characteristic_form", "category"):
            assert example[field] == row[field]
        assert "adaptation_note" not in example


def test_references_exclude_heldout_scenarios_and_asset_identities():
    heldout_ids = {row["id"] for row in HOLDOUT["rows"]}
    heldout_hashes = {row[f"{field}_sha256"] for row in HOLDOUT["rows"]
                      for field in ("text", "characteristic_form")}
    aliases = HOLDOUT["asset_names"] + HOLDOUT["asset_codes"]
    blocked = re.compile(r"\b(?:" + "|".join(re.escape(alias) for alias in aliases) + r")\b", re.I)
    for example in EXAMPLES:
        source = example["source"]
        if source["path"] == HOLDOUT["source_path"]:
            assert source["row_id"] not in heldout_ids
        for field in ("text", "characteristic_form"):
            normalized = re.sub(r"\W+", " ", example[field].casefold()).strip()
            assert hashlib.sha256(normalized.encode()).hexdigest() not in heldout_hashes
            assert not blocked.search(example[field])
    # Other Chiller identities remain legitimate reference examples.
    assert any(re.search(r"\bChiller\b", example["text"], re.I) for example in EXAMPLES)
