import pytest

from scenarios.generation.budget import parse_budget, validate_budget


def test_total_and_case_normalized_plan():
    assert parse_budget(3) == {"scenario_count": 3}
    assert parse_budget(scenario_plan='{"IoT":1,"multi-agent":2}') == {
        "scenario_plan": {"iot": 1, "multiagent": 2}}
    assert parse_budget() == {"scenario_count": 25}


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", {}, {"positive": 1}])
def test_invalid_totals_are_rejected(value):
    with pytest.raises(ValueError):
        parse_budget(value)


@pytest.mark.parametrize("value", ['{}', '{"unknown":1}', '{"iot":0}',
    '{"iot":1,"IoT":2}', '{"iot":1,"iot":2}', '{"iot":{"positive":1}}',
    '{"iot":true}', '{"iot":-1}', 'null', '[]', 'invalid'])
def test_invalid_plans_are_rejected(value):
    with pytest.raises(ValueError):
        parse_budget(scenario_plan=value)


def test_budget_options_are_exclusive():
    with pytest.raises(ValueError):
        parse_budget(1, '{"iot":1}')


def test_total_budget_allows_unused_domains():
    request = parse_budget(2)
    allocation = {"iot": 1, "multiagent": 1}
    scenarios = [{"id": 1, "type": "iot"}, {"id": 2, "type": "multiagent"}]
    assert validate_budget(request, scenarios, allocation) == []
    assert validate_budget(request, scenarios, {"iot": 2})


def test_equal_total_cannot_hide_wrong_domain():
    request = parse_budget(scenario_plan='{"iot":1,"fmsr":1}')
    scenarios = [{"id": 1, "type": "iot"}, {"id": 2, "type": "iot"}]
    errors = validate_budget(request, scenarios, request['scenario_plan'])
    assert 'iot: requested 1, generated 2' in errors
    assert 'fmsr: requested 1, generated 0' in errors
    assert validate_budget(request, [], {"iot": 2})[0] == 'Allocation does not match scenario_plan'


def test_bad_output_cannot_count_toward_a_quota():
    errors = validate_budget(parse_budget(1), [{"id": 1, "type": "invented"}], {"iot": 1})
    assert any('invalid domain' in error for error in errors)
