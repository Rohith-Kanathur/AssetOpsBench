import pytest

from scenarios.generation.budget import parse_budget, validate_budget


def test_counts_and_case_normalized_explicit_plan():
    assert parse_budget('{"positive":3}') == {"scenario_counts": {"positive": 3, "negative": 0}}
    assert parse_budget(scenario_plan='{"IoT":{"positive":1},"multi-agent":{"negative":2}}') == {
        "scenario_plan": {"iot": {"positive": 1, "negative": 0},
                          "multiagent": {"positive": 0, "negative": 2}}}
    assert parse_budget() == {"scenario_counts": {"positive": 50, "negative": 2}}


@pytest.mark.parametrize("value", ['null', '[]', '{}', '{"positive":-1}',
    '{"positive":true}', '{"positive":1.0}', '{"positive":"1"}',
    '{"positive":1,"extra":1}', '{"positive":1,"positive":2}', 'invalid'])
def test_invalid_totals_are_rejected(value):
    with pytest.raises(ValueError):
        parse_budget(value)


@pytest.mark.parametrize("value", ['{}', '{"unknown":{"positive":1}}',
    '{"iot":{"positive":0}}', '{"iot":{"positive":1},"IoT":{"positive":2}}'])
def test_invalid_plans_are_rejected(value):
    with pytest.raises(ValueError):
        parse_budget(scenario_plan=value)


def test_budget_options_are_exclusive():
    with pytest.raises(ValueError):
        parse_budget('{"positive":1}', '{"iot":{"positive":1}}')


def test_total_budget_allows_unused_domains():
    request = parse_budget('{"positive":1,"negative":1}')
    allocation = {"iot": {"positive": 1}, "multiagent": {"negative": 1}}
    scenarios = [{"id": 1, "type": "iot", "positive": True},
                 {"id": 2, "type": "multiagent", "positive": False}]
    assert validate_budget(request, scenarios, allocation) == []
    assert validate_budget(request, scenarios, {"iot": {"positive": 2}})


def test_equal_total_cannot_hide_wrong_polarity_or_domain():
    request = parse_budget(scenario_plan='{"iot":{"positive":1},"fmsr":{"negative":1}}')
    scenarios = [{"id": 1, "type": "iot", "positive": False},
                 {"id": 2, "type": "fmsr", "positive": True}]
    errors = validate_budget(request, scenarios, request['scenario_plan'])
    assert 'iot positive: requested 1, generated 0' in errors
    assert 'fmsr positive: requested 0, generated 1' in errors
    assert validate_budget(request, [], {"iot": {"positive": 2}})[0] == 'Allocation does not match scenario_plan'


def test_bad_output_cannot_count_toward_a_quota():
    request = parse_budget('{"negative":1}')
    for domain, flag in [('invented', False), ('iot', 0)]:
        errors = validate_budget(request, [{"id": 1, "type": domain, "positive": flag}],
                                 {"iot": {"negative": 1}})
        assert any('invalid domain or positive flag' in error for error in errors)
