import functools
import logging

import library.python.retry as lpr


logger = logging.getLogger(__name__)


# TODO: add tests!
def retry(func, *args, **kwargs):
    return lpr.retry_call(func, conf=_conf(func, *args, **kwargs))


def retrying(**retry_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            proxy_func = lambda: func(*args, **kwargs)
            return retry(proxy_func, **retry_kwargs)

        return wrapper

    return decorator


def _conf(
    func,
    max_time=None,
    max_times=None,
    retry_sleep=lambda i, t: i / 10.0,
    raise_exception=lambda e: False,
    ignore_exception=lambda e: True,
    sleep_func=None,
):
    def log_error(error, n, raised_after):
        logger.debug(
            'exception after {t} on {i} try {f}: {c}({r}) ({e})'.format(
                t=raised_after,
                i=n,
                f=func.__name__,
                c=error.__class__.__name__,
                r=repr(error),
                e=error,
            )
        )

    return lpr.RetryConf(
        retriable=lambda e: not raise_exception(e) and ignore_exception(e),
        get_delay=lambda n, raised_after, last: retry_sleep(n, raised_after),
        max_time=max_time,
        max_times=max_times - 1 if max_times else None,  # tries -> retries
        handle_error=log_error,
        logger=None,
        sleep=sleep_func or lpr.DEFAULT_SLEEP_FUNC,
    )
