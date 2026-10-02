"""Generation inventory uses existing data without extending MCP tools."""

from unittest.mock import Mock

import pytest

from scenarios import inventory


def test_asset_coverage_uses_registry_sites_and_timestamp_order(monkeypatch):
    asset_db, iot_db = Mock(), Mock()

    def registry_find(selector, **kwargs):
        if isinstance(selector["siteid"], dict):
            return {"docs": [{"siteid": "NORTH"}, {"siteid": "SOUTH"}]}
        return {"docs": [{"assetnum": f"{selector['siteid']}-1", "assettype": "TRANSFORMER"}]}

    def telemetry_find(selector, **kwargs):
        return {"docs": [
            {"asset_id": selector["asset_id"], "timestamp": "2024-01-01T09:00:00+02:00", "temperature": 20},
            {"asset_id": selector["asset_id"], "timestamp": "2024-01-01T08:00:00+00:00", "pressure": 30},
        ]}

    asset_db.find.side_effect = registry_find
    iot_db.find.side_effect = telemetry_find
    monkeypatch.setattr(inventory.iot, "asset_db", asset_db)
    monkeypatch.setattr(inventory.iot, "iot_db", iot_db)
    monkeypatch.setattr(inventory.iot, "_registry_sites_cache", None)
    monkeypatch.setattr(inventory.iot, "_sensor_list_cache", {})
    rows = inventory.get_asset_coverage()

    assert [(row["site_name"], row["asset_id"]) for row in rows] == [
        ("NORTH", "NORTH-1"), ("SOUTH", "SOUTH-1"),
    ]
    for row in rows:
        assert row["asset_class"] == "TRANSFORMER"
        assert row["sensors"] == ["pressure", "temperature"]
        assert row["time_range"] == {
            "start": "2024-01-01T09:00:00+02:00",
            "end": "2024-01-01T08:00:00+00:00",
            "total_observations": 2,
        }


def test_vibration_coverage_joins_sensor_fields_and_ignores_metadata(monkeypatch):
    database = Mock()
    database.find.return_value = {"docs": [
        {"_id": "later", "asset_id": "Chiller 6", "timestamp": "2024-01-02", "acceleration": 2},
        {"_id": "earlier", "asset_id": "Chiller 6", "timestamp": "2024-01-01", "velocity": 1},
        {"asset_id": "", "timestamp": "2024-01-03", "invalid": 9},
    ]}
    monkeypatch.setattr(inventory.vibration, "_get_db", lambda: database)
    assert inventory.get_vibration_asset_coverage() == [{
        "site_name": "MAIN", "asset_id": "Chiller 6",
        "sensors": ["acceleration", "velocity"],
        "time_range": {"start": "2024-01-01", "end": "2024-01-02", "total_observations": 2},
    }]


@pytest.mark.parametrize("database", [None, Mock()])
def test_unavailable_vibration_does_not_block_iot_grounding(monkeypatch, database):
    if database is not None:
        database.find.side_effect = RuntimeError("Unavailable database")
    monkeypatch.setattr(inventory.vibration, "_get_db", lambda: database)
    assert inventory.get_vibration_asset_coverage() == []
