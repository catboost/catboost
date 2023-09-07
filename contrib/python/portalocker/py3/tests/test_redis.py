import _thread
import logging
import random
import time

import pytest
from redis import client
from redis import exceptions

import portalocker
from portalocker import redis
from portalocker import utils

logger = logging.getLogger(__name__)

try:
    client.Redis().ping()
except (exceptions.ConnectionError, ConnectionRefusedError):
    pytest.skip('Unable to connect to redis', allow_module_level=True)


@pytest.fixture(autouse=True)
def set_redis_timeouts(monkeypatch):
    monkeypatch.setattr(utils, 'DEFAULT_TIMEOUT', 0.0001)
    monkeypatch.setattr(utils, 'DEFAULT_CHECK_INTERVAL', 0.0005)
    monkeypatch.setattr(redis, 'DEFAULT_UNAVAILABLE_TIMEOUT', 0.01)
    monkeypatch.setattr(redis, 'DEFAULT_THREAD_SLEEP_TIME', 0.001)
    monkeypatch.setattr(_thread, 'interrupt_main', lambda: None)


def test_redis_lock():
    channel = str(random.random())

    lock_a = redis.RedisLock(channel)
    lock_a.acquire(fail_when_locked=True)
    time.sleep(0.01)

    lock_b = redis.RedisLock(channel)
    try:
        with pytest.raises(portalocker.AlreadyLocked):
            lock_b.acquire(fail_when_locked=True)
    finally:
        lock_a.release()
        lock_a.connection.close()


@pytest.mark.parametrize('timeout', [None, 0, 0.001])
@pytest.mark.parametrize('check_interval', [None, 0, 0.0005])
def test_redis_lock_timeout(timeout, check_interval):
    connection = client.Redis()
    channel = str(random.random())
    lock_a = redis.RedisLock(channel)
    lock_a.acquire(timeout=timeout, check_interval=check_interval)

    lock_b = redis.RedisLock(channel, connection=connection)
    with pytest.raises(portalocker.AlreadyLocked):
        try:
            lock_b.acquire(timeout=timeout, check_interval=check_interval)
        finally:
            lock_a.release()
            lock_a.connection.close()


def test_redis_lock_context():
    channel = str(random.random())

    lock_a = redis.RedisLock(channel, fail_when_locked=True)
    with lock_a:
        time.sleep(0.01)
        lock_b = redis.RedisLock(channel, fail_when_locked=True)
        with pytest.raises(portalocker.AlreadyLocked):
            with lock_b:
                pass


def test_redis_relock():
    channel = str(random.random())

    lock_a = redis.RedisLock(channel, fail_when_locked=True)
    with lock_a:
        time.sleep(0.01)
        with pytest.raises(AssertionError):
            lock_a.acquire()
    time.sleep(0.01)

    lock_a.release()


if __name__ == '__main__':
    test_redis_lock()
