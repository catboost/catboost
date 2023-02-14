import datetime


def parse_time_delta(s):
    expander = {'s': 'seconds', 'h': 'hours', 'm': 'minutes', 'd': 'days', 'w': 'weeks'}

    value, tail = int(s[:-1]), s[-1]
    kwargs = {expander[tail]: value}
    return datetime.timedelta(**kwargs)
