"""Keep the small example bundle traceable to its existing corpus rows."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scenarios.generation.budget import DOMAINS


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
EXAMPLES = json.loads((PACKAGE / "references/examples.json").read_text())["examples"]


def test_examples_cover_domains_and_retain_exact_source_row_provenance():
    assert Counter(example["type"] for example in EXAMPLES) == {domain: 3 for domain in DOMAINS}
    for example in EXAMPLES:
        source = example["source"]
        path = REPOSITORY / source["path"]
        text = path.read_text()
        rows = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
        row = next(row for row in rows if row["id"] == source["row_id"])
        serialized = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert hashlib.sha256(serialized).hexdigest() == source["sha256"]


def test_example_texts_are_unique_templates_with_operator_and_rubric_separation():
    texts = [example["text"] for example in EXAMPLES]
    assert len(set(texts)) == len(texts)
    original_identifiers = re.compile(r"Chiller \d|CWC\d+|MAIN|Motor_01|PLANT_A|vib_\d+|mp_1|hp_1|20\d{2}-\d{2}-\d{2}")
    for example in EXAMPLES:
        assert "<" in example["text"] and ">" in example["text"]
        assert not original_identifiers.search(example["text"] + example["characteristic_form"])
        assert all(f"{domain}." not in example["text"] for domain in DOMAINS)
        assert isinstance(example["characteristic_form"], str)
        assert f"{example['type']}." in example["characteristic_form"] or example["type"] == "multiagent"
