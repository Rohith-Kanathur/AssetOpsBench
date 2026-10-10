"""Scenario totals and optional allocation across tool domains."""

import json

DOMAINS = ("iot", "fmsr", "tsfm", "wo", "vibration", "multiagent")
DEFAULT_COUNT = 25


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate budget key: {key}")
        result[key] = value
    return result


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Scenario counts must be nonnegative integers")
    return value


def _plan(value):
    if not isinstance(value, dict):
        raise ValueError("Scenario plan must be a JSON object keyed by domain")
    plan = {}
    for name, count in value.items():
        domain = name.lower().replace("multi-agent", "multiagent")
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {name!r}; choose from {', '.join(DOMAINS)}")
        if domain in plan:
            raise ValueError(f"Duplicate domain: {domain}")
        plan[domain] = _count(count)
    return plan


def parse_budget(scenario_count=None, scenario_plan=None):
    """Use one total or an explicit per-domain plan; omitted domains request zero."""
    if scenario_count is not None and scenario_plan is not None:
        raise ValueError("Use either --count or --plan")
    if scenario_plan is not None:
        plan = _plan(json.loads(scenario_plan, object_pairs_hook=_unique_object))
        total = sum(plan.values())
        request = {"scenario_plan": plan}
    else:
        total = _count(scenario_count if scenario_count is not None else DEFAULT_COUNT)
        request = {"scenario_count": total}
    if not total:
        raise ValueError("Request at least one scenario")
    return request


def validate_budget(request, scenarios, allocation):
    """Check the allocation and output against the requested total or domain counts."""
    errors = []
    try:
        count, plan = request.get("scenario_count"), request.get("scenario_plan")
        if (count is None) == (plan is None):
            raise ValueError("Request must contain exactly one scenario budget")
        budget = parse_budget(count, json.dumps(plan) if plan is not None else None)
        planned = _plan(allocation)
        expected = budget.get("scenario_plan")
        if expected is not None:
            if any(planned.get(d, 0) != expected.get(d, 0) for d in DOMAINS):
                errors.append("Allocation does not match scenario_plan")
        elif sum(planned.values()) != budget["scenario_count"]:
            errors.append("Allocation total does not match scenario_count")
    except (ValueError, TypeError, AttributeError) as exc:
        return [f"Invalid scenario budget: {exc}"]
    observed = {domain: 0 for domain in DOMAINS}
    for scenario in scenarios:
        domain = scenario.get("type")
        if domain not in DOMAINS:
            errors.append(f"Scenario {scenario.get('id')}: invalid domain")
            continue
        observed[domain] += 1
    for domain in DOMAINS:
        wanted, actual = planned.get(domain, 0), observed[domain]
        if actual != wanted:
            errors.append(f"{domain}: requested {wanted}, generated {actual}")
    return errors
