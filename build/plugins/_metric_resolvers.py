import re

VALUE_PATTERN = re.compile(r"^\s*(?P<value>\d+)\s*$")


def resolve_value(val):
    match = VALUE_PATTERN.match(val)
    if not match:
        return None
    val = match.group('value')
    return int(val)
