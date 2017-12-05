import re

VALUE_PATTERN = re.compile(r"^\s*(?P<value>\d+)\s*$")
SIZE_METRIC_PATTERN = re.compile(r"(\d*\.?\d+|\d+)(.*)")


def resolve_value(val):
    match = VALUE_PATTERN.match(val)
    if not match:
        return None
    val = match.group('value')
    return int(val)


def resolve_size_metric(val):
    metrics = {
        "gb": 1000 ** 3, "gib": 1024 ** 3,
        "mb": 1000 ** 2, "mib": 1024 ** 2,
    }
    match = SIZE_METRIC_PATTERN.match(val)
    if not match:
        return None

    val, metric = match.groups()
    if not metric:
        return int(val) * 1024 ** 3
    if metric in metrics:
        return int(float(val) * metrics[metric])
    return None
