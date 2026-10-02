"""Collect live asset and sensor coverage for open-form generation."""

import logging

from servers.iot import main as iot
from servers.vibration import couchdb_client as vibration

_log = logging.getLogger(__name__)
_VIBRATION_META_FIELDS = {"_id", "_rev", "asset_id", "timestamp", "site_name"}


def get_asset_coverage() -> list[dict]:
    coverage = []
    for site_name in iot.known_sites():
        registry = iot.assets(site_name)
        if isinstance(registry, iot.ErrorResult):
            _log.warning("Cannot inspect assets at %s: %s", site_name, registry.error)
            continue
        for asset in registry.assets:
            extent = iot.stream_extent(site_name, asset.asset_id)
            if isinstance(extent, iot.ErrorResult):
                _log.warning("Cannot inspect %s at %s: %s", asset.asset_id, site_name, extent.error)
                continue
            coverage.append({
                "site_name": site_name,
                "asset_id": asset.asset_id,
                "asset_class": asset.assettype,
                "sensors": iot.get_sensor_list(asset.asset_id),
                "time_range": {
                    "start": extent.start_time,
                    "end": extent.end_time,
                    "total_observations": extent.total_records,
                },
            })
    return coverage


def get_vibration_asset_coverage() -> list[dict]:
    db = vibration._get_db()
    if db is None:
        return []
    try:
        result = db.find({"asset_id": {"$exists": True}}, limit=100000)
    except Exception as exc:
        _log.warning("Cannot inspect vibration coverage: %s", exc)
        return []

    grouped = {}
    for doc in result.get("docs", []):
        if not isinstance(doc, dict):
            continue
        asset_id = str(doc.get("asset_id", "")).strip()
        if not asset_id:
            continue
        group = grouped.setdefault(asset_id, {"sensors": set(), "timestamps": []})
        group["sensors"].update(key for key in doc if key not in _VIBRATION_META_FIELDS)
        timestamp = doc.get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            group["timestamps"].append(timestamp)

    coverage = []
    for asset_id, group in grouped.items():
        timestamps = sorted(group["timestamps"])
        coverage.append({
            # The benchmark vibration database uses a single MAIN site.
            "site_name": "MAIN",
            "asset_id": asset_id,
            "sensors": sorted(group["sensors"]),
            "time_range": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
                "total_observations": len(timestamps),
            },
        })
    return sorted(coverage, key=lambda row: row["asset_id"].lower())
