import random
import functools


def reservoir_sampling(data, nsamples):
    result = []
    for i, entry in enumerate(data):
        if i < nsamples:
            result.append(entry)
        else:
            j = random.randint(0, i)
            if j < nsamples:
                result[j] = entry
    return result


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
