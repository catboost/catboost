import time
import functools


# Partly copy-pasted from contrib/python/retry
def retry_func(f, exceptions=Exception, tries=-1, delay=1, max_delay=None, backoff=1):
    _tries, _delay = tries, delay
    while _tries:
        try:
            return f()
        except exceptions as e:
            _tries -= 1
            if not _tries:
                raise

            time.sleep(_delay)
            _delay *= backoff

            if max_delay is not None:
                _delay = min(_delay, max_delay)


def retry(**retry_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_func(lambda: func(*args, **kwargs), **retry_kwargs)
        return wrapper
    return decorator
