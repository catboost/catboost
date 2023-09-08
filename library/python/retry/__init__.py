import copy
import datetime
import functools
import itertools
import logging
import random
import time


"""
Retry library provides an ability to retry function calls in a configurable way.

To retry a certain function call use `retry_call` function. To make function auto-retriable use `retry`
or `retry_intrusive` decorators. Both `retry_call` and `retry` optionally accept retry configuration object
or its fields as kwargs. The `retry_intrusive` is designed for methods and uses intrusive configuration object.

>>> retry_call(foo)
>>> retry_call(foo, foo_args, foo_kwargs)
>>> retry_call(foo, foo_args, foo_kwargs, conf=conf)

>>> @retry()
>>> def foo(...):
>>>     ...

>>> @retry(conf)
>>> def foo(...):
>>>     ...

>>> class Obj(object):
>>>     def __init__(self):
>>>         self.retry_conf = conf
>>>
>>>     @retry_intrusive
>>>     def foo(self, ...):
>>>         ...

This library differs from its alternatives:
    * `retry` contrib library lacks time-based limits, reusable configuration objects and is generally less flexible
    * `retrying` contrib library is somewhat more complex, but also lacks reusable configuration objects
"""


DEFAULT_SLEEP_FUNC = time.sleep
LOGGER = logging.getLogger(__name__)


class RetryConf(object):
    """
    Configuration object defines retry behaviour and is composed of these fields:
        * `retriable` - function that decides if an exception should trigger retrying
        * `get_delay` - function that returns a number of seconds retrier must wait before doing the next attempt
        * `max_time` - maximum `datetime.timedelta` that can pass after the first call for any retry attempt to be done
        * `max_times` - maximum number of retry attempts (note retries, not tries/calls)
        * `handle_error` - function that is called for each failed call attempt
        * `logger` - logger object to record retry warnings with
        * `sleep` - custom sleep function to use for waiting

    >>> RetryConf(max_time=datetime.timedelta(seconds=30), max_times=10)

    Empty configuration retries indefinitely on any exceptions raised.

    By default `DEFAULT_CONF` if used, which retries indefinitely, waiting 1 sec with 1.2 backoff between attempts, and
    also logging with built-in logger object.

    Configuration must be cloned before modification to create separate configuration:

    >>> DEFAULT_CONF.clone()

    There are various methods that provide convenient clone-and-modify shortcuts and "retry recipes".
    """

    _PROPS = {
        "retriable": lambda e: True,
        "get_delay": lambda n, raised_after, last: 0,
        "max_time": None,
        "max_times": None,
        "handle_error": None,
        "logger": None,
        "sleep": DEFAULT_SLEEP_FUNC,
    }

    def __init__(self, **kwargs):
        for prop, default_value in self._PROPS.items():
            setattr(self, prop, default_value)
        self._set(**kwargs)

    def __repr__(self):
        return repr(self.__dict__)

    def clone(self, **kwargs):
        """
        Clone configuration.
        """

        obj = copy.copy(self)
        obj._set(**kwargs)
        return obj

    def on(self, *errors):
        """
        Clone and retry on specific exception types (retriable shortcut):

        >>> conf = conf.on(MyException, MyOtherException)
        """

        obj = self.clone()
        obj.retriable = lambda e: isinstance(e, errors)
        return obj

    def waiting(self, delay=0, backoff=1.0, jitter=0, limit=None):
        """
        Clone and wait between attempts with backoff, jitter and limit (get_delay shortcut):

        >>> conf = conf.waiting(delay)
        >>> conf = conf.waiting(delay, backoff=2.0)  # initial delay with backoff x2 on each attempt
        >>> conf = conf.waiting(delay, jitter=3)  # jitter from 0 to 3 seconds
        >>> conf = conf.waiting(delay, backoff=2.0, limit=60)  # delay with backoff, but not greater than a minute

        All these options can be combined together, of course.
        """

        def get_delay(n, raised_after, last):
            if n == 1:
                return delay

            s = last * backoff
            s += random.uniform(0, jitter)
            if limit is not None:
                s = min(s, limit)
            return s

        obj = self.clone()
        obj.get_delay = get_delay
        return obj

    def upto(self, seconds=0, **other_timedelta_kwargs):
        """
        Clone and do retry attempts only for some time (max_time shortcut):

        >>> conf = conf.upto(30)  # retrying for 30 seconds
        >>> conf = conf.upto(hours=1, minutes=20)  # retrying for 1:20

        Any `datetime.timedelta` kwargs can be used here.
        """

        obj = self.clone()
        obj.max_time = datetime.timedelta(seconds=seconds, **other_timedelta_kwargs)
        return obj

    def upto_retries(self, retries=0):
        """
        Set limit for retry attempts number (max_times shortcut):

        >>> conf = conf.upto_retries(10)
        """

        obj = self.clone()
        obj.max_times = retries
        return obj

    def _set(self, **kwargs):
        for prop, value in kwargs.items():
            if prop not in self._PROPS:
                continue
            setattr(self, prop, value)


DEFAULT_CONF = RetryConf(logger=LOGGER).waiting(1, backoff=1.2)


def retry_call(f, f_args=(), f_kwargs={}, conf=DEFAULT_CONF, **kwargs):
    """
    Retry function call.

    :param f:           function to be retried
    :param f_args:      target function args
    :param f_kwargs:    target function kwargs
    :param conf:        configuration
    """

    if kwargs:
        conf = conf.clone(**kwargs)
    return _retry(conf, functools.partial(f, *f_args, **f_kwargs))


def retry(conf=DEFAULT_CONF, **kwargs):
    """
    Retrying decorator.

    :param conf:        configuration
    """

    if kwargs:
        conf = conf.clone(**kwargs)

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            return _retry(conf, functools.partial(f, *f_args, **f_kwargs))

        return wrapped

    return decorator


def retry_intrusive(f):
    """
    Retrying method decorator that uses an intrusive conf (obj.retry_conf).
    """

    @functools.wraps(f)
    def wrapped(obj, *f_args, **f_kwargs):
        assert hasattr(obj, "retry_conf"), "Object must have retry_conf attribute for decorator to run"
        return _retry(obj.retry_conf, functools.partial(f, obj, *f_args, **f_kwargs))

    return wrapped


def _retry(conf, f):
    start = datetime.datetime.now()
    delay = 0
    for n in itertools.count(1):
        try:
            return f()
        except Exception as error:
            raised_after = datetime.datetime.now() - start
            if conf.handle_error:
                conf.handle_error(error, n, raised_after)
            delay = conf.get_delay(n, raised_after, delay)
            retry_after = raised_after + datetime.timedelta(seconds=delay)
            retrying = (
                conf.retriable(error)
                and (conf.max_times is None or n <= conf.max_times)
                and (conf.max_time is None or retry_after <= conf.max_time)
            )
            if not retrying:
                raise
            if delay:
                conf.sleep(delay)
            if conf.logger:
                conf.logger.warning(
                    "Retrying (try %d) after %s (%s + %s sec) on %s: %s",
                    n,
                    retry_after,
                    raised_after,
                    delay,
                    error.__class__.__name__,
                    error,
                    exc_info=True,
                )
