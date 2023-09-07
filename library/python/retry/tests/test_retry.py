import contextlib
import datetime

import pytest

import library.python.retry as retry


def test_default():
    @retry.retry()
    def foo():
        pass

    foo()


def test_exec():
    ctx = {"run": False}

    @retry.retry()
    def foo():
        ctx["run"] = True

    foo()

    assert ctx["run"]


class RetriableError(Exception):
    pass


def test_conf():
    conf = retry.RetryConf()

    conf2 = conf.clone()
    assert conf2 is not conf

    conf_on = conf.on(RetriableError)
    assert conf_on.retriable is not conf.retriable
    assert conf_on.retriable(RetriableError("error"))
    t = datetime.timedelta(seconds=3)

    conf_waiting = conf.waiting(42, backoff=1.5)
    assert conf_waiting.get_delay is not conf.get_delay
    assert conf_waiting.get_delay(3, t, 63) == 94.5


class Counter(object):
    def __init__(self):
        self.value = 0

    def checkin(self):
        self.value += 1


def DUMMY_RUN(*args, **kwargs):
    return None


@contextlib.contextmanager
def erroneous_runner(run, n=1, error=Exception):
    counter = Counter()

    def wrapped_run(*args, **kwargs):
        counter.checkin()
        if counter.value <= n:
            raise error("Error")
        return run(*args, **kwargs)

    yield wrapped_run


@contextlib.contextmanager
def counting_runner(run, counter):
    def wrapped_run(*args, **kwargs):
        counter.checkin()
        return run(*args, **kwargs)

    yield wrapped_run


param_runs = pytest.mark.parametrize("runs", (1, 2, 3))


@param_runs
def test_retries_call(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            def foo():
                run()

    retry.retry_call(foo)
    assert counter.value == runs + 1


@param_runs
def test_retries_call_args(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            def foo(arg, kwarg=None):
                import logging

                logging.info("!!! %s %s", arg, kwarg)
                run()

    retry.retry_call(foo, (1,), {"kwarg": 2})
    assert counter.value == runs + 1


@param_runs
def test_retries_decorator(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(retry.RetryConf())
            def foo():
                run()

    foo()
    assert counter.value == runs + 1


@param_runs
def test_retries_decorator_args(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(retry.RetryConf())
            def foo(arg, kwarg=None):
                run()

    foo(1, kwarg=2)
    assert counter.value == runs + 1


@param_runs
def test_retries_decorator_method(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            class Bar(object):
                @retry.retry(retry.RetryConf())
                def foo(self):
                    run()

    Bar().foo()
    assert counter.value == runs + 1


@param_runs
def test_retries_decorator_method_args(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            class Bar(object):
                @retry.retry(retry.RetryConf())
                def foo(self, arg, kwarg=None):
                    run()

    Bar().foo(1, kwarg=2)
    assert counter.value == runs + 1


@param_runs
def test_retries_decorator_intrusive(runs):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, runs) as run:
        with counting_runner(run, counter) as run:

            class Bar(object):
                def __init__(self):
                    self.retry_conf = retry.RetryConf()

                @retry.retry_intrusive
                def foo(self, arg, kwarg=None):
                    run()

    Bar().foo(1, kwarg=2)
    assert counter.value == runs + 1


def test_retries_decorator_intrusive_fail():
    class Bar(object):
        @retry.retry_intrusive
        def foo(self, arg, kwarg=None):
            pass

    with pytest.raises(AssertionError):
        Bar().foo(1, kwarg=2)


@pytest.mark.parametrize(
    "conf",
    (
        retry.RetryConf(),
        retry.DEFAULT_CONF,
    ),
)
def test_confs(conf):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN) as run:
        with counting_runner(run, counter) as run:

            def foo():
                run()

    retry.retry_call(foo, conf=conf)
    assert counter.value == 2

    counter = Counter()
    with erroneous_runner(DUMMY_RUN) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo_retried():
                run()

    foo_retried()
    assert counter.value == 2


@pytest.mark.parametrize(
    "conf",
    (
        retry.RetryConf().on(RetriableError),
        retry.RetryConf(retriable=lambda e: isinstance(e, RetriableError)),
    ),
)
def test_retriable(conf):
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, error=RetriableError) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    foo()
    assert counter.value == 2

    counter = Counter()
    with erroneous_runner(DUMMY_RUN) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    with pytest.raises(Exception):
        foo()
    assert counter.value == 1


def test_waiting():
    conf = retry.RetryConf().waiting(1)
    with erroneous_runner(DUMMY_RUN) as run:

        @retry.retry(conf)
        def foo():
            run()

    foo()


def test_waiting_backoff():
    conf = retry.RetryConf().waiting(1, backoff=2)
    with erroneous_runner(DUMMY_RUN) as run:

        @retry.retry(conf)
        def foo():
            run()

    foo()


def test_waiting_jitter():
    conf = retry.RetryConf().waiting(0, jitter=1)
    with erroneous_runner(DUMMY_RUN) as run:

        @retry.retry(conf)
        def foo():
            run()

    foo()


def test_upto():
    conf = retry.RetryConf().upto(0)

    counter = Counter()
    with erroneous_runner(DUMMY_RUN) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    with pytest.raises(Exception):
        foo()
    assert counter.value == 1


def test_upto_retries():
    conf = retry.RetryConf().upto_retries(0)
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, 2) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    with pytest.raises(Exception):
        foo()
    assert counter.value == 1

    conf = retry.RetryConf().upto_retries(1)
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, 2) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    with pytest.raises(Exception):
        foo()
    assert counter.value == 2

    conf = retry.RetryConf().upto_retries(2)
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, 2) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    foo()
    assert counter.value == 3

    conf = retry.RetryConf().upto_retries(4)
    counter = Counter()
    with erroneous_runner(DUMMY_RUN, 2) as run:
        with counting_runner(run, counter) as run:

            @retry.retry(conf)
            def foo():
                run()

    foo()
    assert counter.value == 3
