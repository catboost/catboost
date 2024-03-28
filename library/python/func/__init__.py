import functools
import threading
import collections
import contextlib
import six


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

    lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return result.result
        except AttributeError:
            with lock:
                try:
                    return result.result
                except AttributeError:
                    result.result = func(*args, **kwargs)

            return result.result

    return wrapper


def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    lock = threading.Lock()

    @property
    def _lazy_property(self):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)

        with lock:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, fn(self))
            return getattr(self, attr_name)

    return _lazy_property


class classproperty(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        return self.func(owner)


class lazy_classproperty(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        attr_name = '_lazy_' + self.func.__name__

        if not hasattr(owner, attr_name):
            setattr(owner, attr_name, self.func(owner))
        return getattr(owner, attr_name)


class nullcontext(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def memoize(limit=0, thread_local=False, thread_safe=True):
    assert limit >= 0
    assert limit <= 0 or thread_safe, 'memoize() it not thread safe enough to work in limiting and non-thread safe mode'

    def decorator(func):
        memory = {}

        if six.PY3:
            lock = contextlib.nullcontext()
        else:
            lock = nullcontext()
        lock = threading.Lock() if thread_safe else lock

        if limit:
            keys = collections.deque()

            def get(args):
                if args not in memory:
                    with lock:
                        if args not in memory:
                            fargs = args[-1]
                            memory[args] = func(*fargs)
                            keys.append(args)
                            if len(keys) > limit:
                                del memory[keys.popleft()]
                return memory[args]

        else:

            def get(args):
                if args not in memory:
                    with lock:
                        if args not in memory:
                            fargs = args[-1]
                            memory.setdefault(args, func(*fargs))
                return memory[args]

        if thread_local:

            @functools.wraps(func)
            def wrapper(*args):
                th = threading.current_thread()
                return get((th.ident, th.name, args))

        else:

            @functools.wraps(func)
            def wrapper(*args):
                return get(('', '', args))

        return wrapper

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


def first(it):
    for d in it:
        if d:
            return d


def split(data, func):
    l, r = [], []
    for e in data:
        if func(e):
            l.append(e)
        else:
            r.append(e)
    return l, r


def flatten_dict(dd, separator='.', prefix=''):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )
