"""Require lineage to observed records or policy-authorized existing fixtures."""

from .contracts import string, strings


def validate_data_sources(sources, fixture_files=None):
    errors = []
    records = {s["id"]: s for s in sources if isinstance(s, dict) and string(s.get("id"))}
    data = {sid: s for sid, s in records.items() if s.get("role") == "data"}
    usable = set()
    for sid, source in data.items():
        label = f"Data source {sid}"
        kind = source.get("kind")
        if kind in {"observed", "fixture"}:
            if kind == "fixture" and fixture_files is None:
                errors.append(f"{label}: fixture roots require the existing environment policy")
                continue
            if fixture_files is not None:
                files = source.get("files", [])
                evidence = files.items() if isinstance(files, dict) else (
                    (f.get("path"), f.get("sha256")) for f in files if isinstance(f, dict)
                ) if isinstance(files, list) else []
                if not any(isinstance(path, str) and path in fixture_files and fixture_files[path] == digest
                           for path, digest in evidence):
                    errors.append(f"{label}: data root must reference a checksummed existing environment file")
                    continue
            if source.get("input_source_ids"):
                errors.append(f"{label}: transformed records must be marked derived or synthetic")
            else:
                usable.add(sid)
            continue
        if not string(kind) or kind not in {"derived", "simulated", "synthetic"}:
            errors.append(f"{label}: invalid data kind")
            continue
        if fixture_files is not None and kind != "derived":
            errors.append(f"{label}: existing environment permits only lossless derived representations")
            continue
        inputs = source.get("input_source_ids")
        if not strings(inputs, True) or not set(inputs) <= data.keys():
            errors.append(f"{label}: input_source_ids must reference retained data, not literature or receipts")
            continue
        transform = source.get("transform")
        if not isinstance(transform, dict):
            errors.append(f"{label}: a reproducible transform is required")
            continue
        missing = [k for k in ("script", "description", "assumptions", "limitations") if not string(transform.get(k))]
        mapping = transform.get("field_mapping")
        if not isinstance(mapping, dict) or not mapping or not all(string(k) and string(v) for k, v in mapping.items()):
            missing.append("field_mapping")
        files = source.get("files", [])
        paths = set(files) if isinstance(files, dict) else {
            f.get("path") for f in files if isinstance(f, dict) and string(f.get("path"))
        } if isinstance(files, list) else set()
        if not string(transform.get("script")) or transform["script"] not in paths:
            missing.append("script in checksummed files")
        if missing:
            errors.append(f"{label}: transform requires {', '.join(missing)}")
        else:
            usable.add(sid)

    # Resolve from observed roots, so cycles and chains ending in invented data fail.
    grounded = {sid for sid in usable if data[sid]["kind"] in {"observed", "fixture"}}
    while True:
        found = {sid for sid in usable - grounded
                 if set(data[sid]["input_source_ids"]) <= grounded}
        if not found:
            break
        grounded.update(found)
    for sid in data.keys() - grounded:
        errors.append(f"Data source {sid}: no complete lineage to observed data")
    return errors, grounded


def validate_data_references(profile, grounded, scenarios=()):
    errors = []

    def require(value, label):
        refs = value.get("data_source_ids")
        if not strings(refs, True) or not set(refs) <= grounded:
            errors.append(f"{label}: data_source_ids must reference data grounded in observed records")

    assets = profile.get("assets", [])
    for index, asset in enumerate(assets if isinstance(assets, list) else []):
        if not isinstance(asset, dict):
            continue
        label = f"Profile asset {index}"
        require(asset, label)
        for domain in ("iot", "vibration"):
            coverage = asset.get(domain)
            if isinstance(coverage, dict) and coverage.get("sensors"):
                require(coverage, f"{label}.{domain}")
    for scenario in scenarios:
        ground = scenario.get("grounding")
        if isinstance(ground, dict) and ground.get("scope") == "asset":
            require(ground, f"Scenario {scenario.get('id')}")
    return errors
