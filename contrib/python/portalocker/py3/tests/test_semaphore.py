import random
import pytest
import portalocker
from portalocker import utils


@pytest.mark.parametrize('timeout', [None, 0, 0.001])
@pytest.mark.parametrize('check_interval', [None, 0, 0.0005])
def test_bounded_semaphore(timeout, check_interval, monkeypatch):
    n = 2
    name = random.random()
    monkeypatch.setattr(utils, 'DEFAULT_TIMEOUT', 0.0001)
    monkeypatch.setattr(utils, 'DEFAULT_CHECK_INTERVAL', 0.0005)

    semaphore_a = portalocker.BoundedSemaphore(n, name=name, timeout=timeout)
    semaphore_b = portalocker.BoundedSemaphore(n, name=name, timeout=timeout)
    semaphore_c = portalocker.BoundedSemaphore(n, name=name, timeout=timeout)

    semaphore_a.acquire(timeout=timeout)
    semaphore_b.acquire()
    with pytest.raises(portalocker.AlreadyLocked):
        semaphore_c.acquire(check_interval=check_interval, timeout=timeout)
