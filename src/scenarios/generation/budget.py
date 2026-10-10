"""Positive and negative quotas for automatic allocation or an explicit domain plan."""

import json

DOMAINS = ("iot", "fmsr", "tsfm", "wo", "vibration", "multiagent")
DEFAULT_COUNTS = {"positive": 20, "negative": 5}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate budget key: {key}")
        result[key] = value
    return result


def _counts(value):
    if not isinstance(value, dict) or set(value) - {"positive", "negative"}:
        raise ValueError("Counts must be an object with positive and/or negative keys")
    counts = {kind: value.get(kind, 0) for kind in DEFAULT_COUNTS}
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Scenario counts must be nonnegative integers")
    return counts


def _plan(value):
    if not isinstance(value, dict):
        raise ValueError("Scenario plan must be a JSON object keyed by domain")
    plan = {}
    for name, counts in value.items():
        domain = name.lower().replace("multi-agent", "multiagent")
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {name!r}; choose from {', '.join(DOMAINS)}")
        if domain in plan:
            raise ValueError(f"Duplicate domain: {domain}")
        plan[domain] = _counts(counts)
    return plan


def _totals(plan):
    return {kind: sum(counts[kind] for counts in plan.values()) for kind in DEFAULT_COUNTS}


def parse_budget(scenario_counts=None, scenario_plan=None):
    """Parse CLI JSON; missing domains or polarity counts request zero."""
    if scenario_counts is not None and scenario_plan is not None:
        raise ValueError("Use either --counts or --plan")
    if scenario_plan is not None:
        plan = _plan(json.loads(scenario_plan, object_pairs_hook=_unique_object))
        totals = _totals(plan)
        request = {"scenario_plan": plan}
    else:
        totals = _counts(json.loads(scenario_counts, object_pairs_hook=_unique_object)) if scenario_counts is not None else dict(DEFAULT_COUNTS)
        request = {"scenario_counts": totals}
    if not sum(totals.values()):
        raise ValueError("Request at least one positive or negative scenario")
    return request


def validate_budget(request, scenarios, allocation):
    """Check allocation and actual outputs independently against the requested quotas."""
    errors = []
    try:
        counts, plan = request.get("scenario_counts"), request.get("scenario_plan")
        if (counts is None) == (plan is None):
            raise ValueError("Request must contain exactly one scenario budget")
        budget = parse_budget(json.dumps(counts) if counts is not None else None,
                              json.dumps(plan) if plan is not None else None)
        planned = _plan(allocation)
        expected = budget.get("scenario_plan")
        if expected is not None:
            zero = {kind: 0 for kind in DEFAULT_COUNTS}
            if any(planned.get(d, zero) != expected.get(d, zero) for d in DOMAINS):
                errors.append("Allocation does not match scenario_plan")
        elif _totals(planned) != budget["scenario_counts"]:
            errors.append("Allocation totals do not match scenario_counts")
    except (ValueError, TypeError, AttributeError) as exc:
        return [f"Invalid scenario budget: {exc}"]
    observed = {domain: {kind: 0 for kind in DEFAULT_COUNTS} for domain in DOMAINS}
    for scenario in scenarios:
        domain, positive = scenario.get("type"), scenario.get("positive")
        if domain not in DOMAINS or type(positive) is not bool:
            errors.append(f"Scenario {scenario.get('id')}: invalid domain or positive flag")
            continue
        observed[domain]["positive" if positive else "negative"] += 1
    for domain in DOMAINS:
        for kind in DEFAULT_COUNTS:
            wanted = planned.get(domain, {}).get(kind, 0)
            actual = observed[domain][kind]
            if actual != wanted:
                errors.append(f"{domain} {kind}: requested {wanted}, generated {actual}")
    return errors
