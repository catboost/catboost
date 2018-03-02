import functools
import threading
import collections


def map0(func, value):
    return func(value) if value is not None else value


def single(x):
    if len(x) != 1:
        raise Exception('Length of {} is not equal to 1'.format(x))
    return x[0]


class _Result(object):
    pass


def lazy(func):
    result = _Result()

    @functools.wraps(func)
    def wrapper(*args):
        try:
            return result.result
        except AttributeError:
            result.result = func(*args)

        return result.result

    return wrapper


def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def memoize(thread_safe=False, limit=0):
    assert limit >= 0

    def decorator(func):
        @functools.wraps(func)
        def wrapper_with_memory(memory, lock, keys):
            # remove branching for options
            if limit:
                def get(args):
                    if args not in memory:
                        memory[args] = func(*args)
                        keys.append(args)
                        if len(keys) > limit:
                            del memory[keys.popleft()]
                    return memory[args]
            else:
                def get(args):
                    if args not in memory:
                        memory[args] = func(*args)
                    return memory[args]

            if thread_safe:
                def wrapper(*args):
                    with lock:
                        return get(args)
            else:
                def wrapper(*args):
                    return get(args)

            return wrapper
        return wrapper_with_memory({}, threading.Lock() if thread_safe else None, collections.deque() if limit else None)
    return decorator


# XXX: add test
def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def stable_uniq(it):
    seen = set()
    res = []
    for e in it:
        if e not in seen:
            res.append(e)
            seen.add(e)
    return res
