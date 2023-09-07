import os
import time
import atexit
import tempfile
import contextlib
from . import exceptions
from . import constants
from . import portalocker

current_time = getattr(time, "monotonic", time.time)

DEFAULT_TIMEOUT = 5
DEFAULT_CHECK_INTERVAL = 0.25
LOCK_METHOD = constants.LOCK_EX | constants.LOCK_NB

__all__ = [
    'Lock',
    'open_atomic',
]


@contextlib.contextmanager
def open_atomic(filename, binary=True):
    '''Open a file for atomic writing. Instead of locking this method allows
    you to write the entire file and move it to the actual location. Note that
    this makes the assumption that a rename is atomic on your platform which
    is generally the case but not a guarantee.

    http://docs.python.org/library/os.html#os.rename

    >>> filename = 'test_file.txt'
    >>> if os.path.exists(filename):
    ...     os.remove(filename)

    >>> with open_atomic(filename) as fh:
    ...     written = fh.write(b'test')
    >>> assert os.path.exists(filename)
    >>> os.remove(filename)

    '''
    assert not os.path.exists(filename), '%r exists' % filename
    path, name = os.path.split(filename)

    # Create the parent directory if it doesn't exist
    if path and not os.path.isdir(path):  # pragma: no cover
        os.makedirs(path)

    temp_fh = tempfile.NamedTemporaryFile(
        mode=binary and 'wb' or 'w',
        dir=path,
        delete=False,
    )
    yield temp_fh
    temp_fh.flush()
    os.fsync(temp_fh.fileno())
    temp_fh.close()
    try:
        os.rename(temp_fh.name, filename)
    finally:
        try:
            os.remove(temp_fh.name)
        except Exception:
            pass


class Lock(object):

    def __init__(
            self, filename, mode='a', timeout=DEFAULT_TIMEOUT,
            check_interval=DEFAULT_CHECK_INTERVAL, fail_when_locked=False,
            flags=LOCK_METHOD, **file_open_kwargs):
        '''Lock manager with build-in timeout

        filename -- filename
        mode -- the open mode, 'a' or 'ab' should be used for writing
        truncate -- use truncate to emulate 'w' mode, None is disabled, 0 is
            truncate to 0 bytes
        timeout -- timeout when trying to acquire a lock
        check_interval -- check interval while waiting
        fail_when_locked -- after the initial lock failed, return an error
            or lock the file
        **file_open_kwargs -- The kwargs for the `open(...)` call

        fail_when_locked is useful when multiple threads/processes can race
        when creating a file. If set to true than the system will wait till
        the lock was acquired and then return an AlreadyLocked exception.

        Note that the file is opened first and locked later. So using 'w' as
        mode will result in truncate _BEFORE_ the lock is checked.
        '''

        if 'w' in mode:
            truncate = True
            mode = mode.replace('w', 'a')
        else:
            truncate = False

        self.fh = None
        self.filename = filename
        self.mode = mode
        self.truncate = truncate
        self.timeout = timeout
        self.check_interval = check_interval
        self.fail_when_locked = fail_when_locked
        self.flags = flags
        self.file_open_kwargs = file_open_kwargs

    def acquire(
            self, timeout=None, check_interval=None, fail_when_locked=None):
        '''Acquire the locked filehandle'''
        if timeout is None:
            timeout = self.timeout
        if timeout is None:
            timeout = 0

        if check_interval is None:
            check_interval = self.check_interval

        if fail_when_locked is None:
            fail_when_locked = self.fail_when_locked

        # If we already have a filehandle, return it
        fh = self.fh
        if fh:
            return fh

        # Get a new filehandler
        fh = self._get_fh()
        try:
            # Try to lock
            fh = self._get_lock(fh)
        except exceptions.LockException as exception:
            # Try till the timeout has passed
            timeoutend = current_time() + timeout
            while timeoutend > current_time():
                # Wait a bit
                time.sleep(check_interval)

                # Try again
                try:

                    # We already tried to the get the lock
                    # If fail_when_locked is true, then stop trying
                    if fail_when_locked:
                        raise exceptions.AlreadyLocked(exception)

                    else:  # pragma: no cover
                        # We've got the lock
                        fh = self._get_lock(fh)
                        break

                except exceptions.LockException:
                    pass

            else:
                fh.close()
                # We got a timeout... reraising
                raise exceptions.LockException(exception)

        # Prepare the filehandle (truncate if needed)
        fh = self._prepare_fh(fh)

        self.fh = fh
        return fh

    def release(self):
        '''Releases the currently locked file handle'''
        if self.fh:
            portalocker.unlock(self.fh)
            self.fh.close()
            self.fh = None

    def _get_fh(self):
        '''Get a new filehandle'''
        return open(self.filename, self.mode, **self.file_open_kwargs)

    def _get_lock(self, fh):
        '''
        Try to lock the given filehandle

        returns LockException if it fails'''
        portalocker.lock(fh, self.flags)
        return fh

    def _prepare_fh(self, fh):
        '''
        Prepare the filehandle for usage

        If truncate is a number, the file will be truncated to that amount of
        bytes
        '''
        if self.truncate:
            fh.seek(0)
            fh.truncate(0)

        return fh

    def __enter__(self):
        return self.acquire()

    def __exit__(self, type_, value, tb):
        self.release()

    def __delete__(self, instance):  # pragma: no cover
        instance.release()


class RLock(Lock):
    """
    A reentrant lock, functions in a similar way to threading.RLock in that it
    can be acquired multiple times.  When the corresponding number of release()
    calls are made the lock will finally release the underlying file lock.
    """
    def __init__(
            self, filename, mode='a', timeout=DEFAULT_TIMEOUT,
            check_interval=DEFAULT_CHECK_INTERVAL, fail_when_locked=False,
            flags=LOCK_METHOD):
        super(RLock, self).__init__(filename, mode, timeout, check_interval,
                                    fail_when_locked, flags)
        self._acquire_count = 0

    def acquire(
            self, timeout=None, check_interval=None, fail_when_locked=None):
        if self._acquire_count >= 1:
            fh = self.fh
        else:
            fh = super(RLock, self).acquire(timeout, check_interval,
                                            fail_when_locked)
        self._acquire_count += 1
        return fh

    def release(self):
        if self._acquire_count == 0:
            raise exceptions.LockException(
                "Cannot release more times than acquired")

        if self._acquire_count == 1:
            super(RLock, self).release()
        self._acquire_count -= 1


class TemporaryFileLock(Lock):

    def __init__(self, filename='.lock', timeout=DEFAULT_TIMEOUT,
                 check_interval=DEFAULT_CHECK_INTERVAL, fail_when_locked=True,
                 flags=LOCK_METHOD):

        Lock.__init__(self, filename=filename, mode='w', timeout=timeout,
                      check_interval=check_interval,
                      fail_when_locked=fail_when_locked, flags=flags)
        atexit.register(self.release)

    def release(self):
        Lock.release(self)
        if os.path.isfile(self.filename):  # pragma: no branch
            os.unlink(self.filename)
