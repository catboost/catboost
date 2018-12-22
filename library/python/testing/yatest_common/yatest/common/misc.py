import functools


def first(it):
    for d in it:
        if d:
            return d


def lazy(func):
    res = []

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not res:
            res.append(func(*args, **kwargs))
        return res[0]

    return wrapper
