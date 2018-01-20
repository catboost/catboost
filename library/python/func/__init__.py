import functools
import threading


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


def memoize(thread_safe=False):

    def decorator(func):

        @functools.wraps(func)
        def wrapper_with_memory(memory, lock):

            def get(args):
                if args not in memory:
                    memory[args] = func(*args)
                return memory[args]

            def wrapper(*args):
                if lock:
                    with lock:
                        return get(args)
                return get(args)

            return wrapper
        return wrapper_with_memory({}, threading.Lock() if thread_safe else None)
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
