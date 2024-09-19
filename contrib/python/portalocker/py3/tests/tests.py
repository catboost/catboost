from __future__ import print_function
from __future__ import with_statement

import os
import dataclasses
import multiprocessing
import time
import typing

import pytest
import portalocker
from portalocker import utils
from portalocker import LockFlags


def test_exceptions(tmpfile):
    # Open the file 2 times
    a = open(tmpfile, 'a')
    b = open(tmpfile, 'a')

    # Lock exclusive non-blocking
    lock_flags = portalocker.LOCK_EX | portalocker.LOCK_NB

    # First lock file a
    portalocker.lock(a, lock_flags)

    # Now see if we can lock file b
    with pytest.raises(portalocker.LockException):
        portalocker.lock(b, lock_flags)

    # Cleanup
    a.close()
    b.close()


def test_utils_base():
    class Test(utils.LockBase):
        pass


def test_with_timeout(tmpfile):
    # Open the file 2 times
    with pytest.raises(portalocker.AlreadyLocked):
        with portalocker.Lock(tmpfile, timeout=0.1) as fh:
            print('writing some stuff to my cache...', file=fh)
            with portalocker.Lock(
                tmpfile, timeout=0.1, mode='wb',
                fail_when_locked=True
            ):
                pass
            print('writing more stuff to my cache...', file=fh)


def test_without_timeout(tmpfile):
    # Open the file 2 times
    with pytest.raises(portalocker.LockException):
        with portalocker.Lock(tmpfile, timeout=None) as fh:
            print('writing some stuff to my cache...', file=fh)
            with portalocker.Lock(tmpfile, timeout=None, mode='w'):
                pass
            print('writing more stuff to my cache...', file=fh)


def test_without_fail(tmpfile):
    # Open the file 2 times
    with pytest.raises(portalocker.LockException):
        with portalocker.Lock(tmpfile, timeout=0.1) as fh:
            print('writing some stuff to my cache...', file=fh)
            lock = portalocker.Lock(tmpfile, timeout=0.1)
            lock.acquire(check_interval=0.05, fail_when_locked=False)


def test_simple(tmpfile):
    with open(tmpfile, 'w') as fh:
        fh.write('spam and eggs')

    fh = open(tmpfile, 'r+')
    portalocker.lock(fh, portalocker.LOCK_EX)

    fh.seek(13)
    fh.write('foo')

    # Make sure we didn't overwrite the original text
    fh.seek(0)
    assert fh.read(13) == 'spam and eggs'

    portalocker.unlock(fh)
    fh.close()


def test_truncate(tmpfile):
    with open(tmpfile, 'w') as fh:
        fh.write('spam and eggs')

    with portalocker.Lock(tmpfile, mode='a+') as fh:
        # Make sure we didn't overwrite the original text
        fh.seek(0)
        assert fh.read(13) == 'spam and eggs'

    with portalocker.Lock(tmpfile, mode='w+') as fh:
        # Make sure we truncated the file
        assert fh.read() == ''


def test_class(tmpfile):
    lock = portalocker.Lock(tmpfile)
    lock2 = portalocker.Lock(tmpfile, fail_when_locked=False, timeout=0.01)

    with lock:
        lock.acquire()

        with pytest.raises(portalocker.LockException):
            with lock2:
                pass

    with lock2:
        pass


def test_acquire_release(tmpfile):
    lock = portalocker.Lock(tmpfile)
    lock2 = portalocker.Lock(tmpfile, fail_when_locked=False)

    lock.acquire()  # acquire lock when nobody is using it
    with pytest.raises(portalocker.LockException):
        # another party should not be able to acquire the lock
        lock2.acquire(timeout=0.01)

        # re-acquire a held lock is a no-op
        lock.acquire()

    lock.release()  # release the lock
    lock.release()  # second release does nothing


def test_rlock_acquire_release_count(tmpfile):
    lock = portalocker.RLock(tmpfile)
    # Twice acquire
    h = lock.acquire()
    assert not h.closed
    lock.acquire()
    assert not h.closed

    # Two release
    lock.release()
    assert not h.closed
    lock.release()
    assert h.closed


def test_rlock_acquire_release(tmpfile):
    lock = portalocker.RLock(tmpfile)
    lock2 = portalocker.RLock(tmpfile, fail_when_locked=False)

    lock.acquire()  # acquire lock when nobody is using it
    with pytest.raises(portalocker.LockException):
        # another party should not be able to acquire the lock
        lock2.acquire(timeout=0.01)

    # Now acquire again
    lock.acquire()

    lock.release()  # release the lock
    lock.release()  # second release does nothing


def test_release_unacquired(tmpfile):
    with pytest.raises(portalocker.LockException):
        portalocker.RLock(tmpfile).release()


def test_exlusive(tmpfile):
    with open(tmpfile, 'w') as fh:
        fh.write('spam and eggs')

    fh = open(tmpfile, 'r')
    portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)

    # Make sure we can't read the locked file
    with pytest.raises(portalocker.LockException):
        with open(tmpfile, 'r') as fh2:
            portalocker.lock(fh2, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh2.read()

    # Make sure we can't write the locked file
    with pytest.raises(portalocker.LockException):
        with open(tmpfile, 'w+') as fh2:
            portalocker.lock(fh2, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh2.write('surprise and fear')

    # Make sure we can explicitly unlock the file
    portalocker.unlock(fh)
    fh.close()


def test_shared(tmpfile):
    with open(tmpfile, 'w') as fh:
        fh.write('spam and eggs')

    f = open(tmpfile, 'r')
    portalocker.lock(f, portalocker.LOCK_SH | portalocker.LOCK_NB)

    # Make sure we can read the locked file
    with open(tmpfile, 'r') as fh2:
        portalocker.lock(fh2, portalocker.LOCK_SH | portalocker.LOCK_NB)
        assert fh2.read() == 'spam and eggs'

    # Make sure we can't write the locked file
    with pytest.raises(portalocker.LockException):
        with open(tmpfile, 'w+') as fh2:
            portalocker.lock(fh2, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh2.write('surprise and fear')

    # Make sure we can explicitly unlock the file
    portalocker.unlock(f)
    f.close()


def test_blocking_timeout(tmpfile):
    flags = LockFlags.SHARED

    with pytest.warns(UserWarning):
        with portalocker.Lock(tmpfile, timeout=5, flags=flags):
            pass

    lock = portalocker.Lock(tmpfile, flags=flags)
    with pytest.warns(UserWarning):
        lock.acquire(timeout=5)


@pytest.mark.skipif(os.name == 'nt',
                    reason='Windows uses an entirely different lockmechanism')
def test_nonblocking(tmpfile):
    with open(tmpfile, 'w') as fh:
        with pytest.raises(RuntimeError):
            portalocker.lock(fh, LockFlags.NON_BLOCKING)


def shared_lock(filename, **kwargs):
    with portalocker.Lock(
        filename,
        timeout=0.1,
        fail_when_locked=False,
        flags=LockFlags.SHARED | LockFlags.NON_BLOCKING,
    ):
        time.sleep(0.2)
        return True


def shared_lock_fail(filename, **kwargs):
    with portalocker.Lock(
        filename,
        timeout=0.1,
        fail_when_locked=True,
        flags=LockFlags.SHARED | LockFlags.NON_BLOCKING,
    ):
        time.sleep(0.2)
        return True


def exclusive_lock(filename, **kwargs):
    with portalocker.Lock(
        filename,
        timeout=0.1,
        fail_when_locked=False,
        flags=LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING,
    ):
        time.sleep(0.2)
        return True


@dataclasses.dataclass(order=True)
class LockResult:
    exception_class: typing.Union[typing.Type[Exception], None] = None
    exception_message: typing.Union[str, None] = None
    exception_repr: typing.Union[str, None] = None


def lock(
    filename: str,
    fail_when_locked: bool,
    flags: LockFlags
) -> LockResult:
    # Returns a case of True, False or FileNotFound
    # https://thedailywtf.com/articles/what_is_truth_0x3f_
    # But seriously, the exception properties cannot be safely pickled so we
    # only return string representations of the exception properties
    try:
        with portalocker.Lock(
            filename,
            timeout=0.1,
            fail_when_locked=fail_when_locked,
            flags=flags,
        ):
            time.sleep(0.2)
            return LockResult()

    except Exception as exception:
        # The exceptions cannot be pickled so we cannot return them through
        # multiprocessing
        return LockResult(
            type(exception),
            str(exception),
            repr(exception),
        )


@pytest.mark.parametrize('fail_when_locked', [True, False])
def test_shared_processes(tmpfile, fail_when_locked):
    flags = LockFlags.SHARED | LockFlags.NON_BLOCKING

    with multiprocessing.Pool(processes=2) as pool:
        args = tmpfile, fail_when_locked, flags
        results = pool.starmap_async(lock, 2 * [args])

        for result in results.get(timeout=3):
            assert result == LockResult()


@pytest.mark.parametrize('fail_when_locked', [True, False])
def test_exclusive_processes(tmpfile, fail_when_locked):
    flags = LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING

    with multiprocessing.Pool(processes=2) as pool:
        # filename, fail_when_locked, flags
        args = tmpfile, fail_when_locked, flags
        a, b = pool.starmap_async(lock, 2 * [args]).get(timeout=3)

        assert not a.exception_class or not b.exception_class
        assert issubclass(
            a.exception_class or b.exception_class,
            portalocker.LockException
        )


@pytest.mark.skipif(
    os.name == 'nt',
    reason='Locking on Windows requires a file object',
)
def test_lock_fileno(tmpfile):
    # Open the file 2 times
    a = open(tmpfile, 'a')
    b = open(tmpfile, 'a')

    # Lock exclusive non-blocking
    flags = LockFlags.SHARED | LockFlags.NON_BLOCKING

    # First lock file a
    portalocker.lock(a, flags)

    # Now see if we can lock using fileno()
    portalocker.lock(b.fileno(), flags)

    # Cleanup
    a.close()
    b.close()

