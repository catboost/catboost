import logging
import time
from threading import Lock

from ._error import Timeout

_LOGGER = logging.getLogger(__name__)


# This is a helper class which is returned by :meth:`BaseFileLock.acquire` and wraps the lock to make sure __enter__
# is not called twice when entering the with statement. If we would simply return *self*, the lock would be acquired
# again in the *__enter__* method of the BaseFileLock, but not released again automatically. issue #37 (memory leak)
class AcquireReturnProxy(object):
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        return self.lock

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: U100
        self.lock.release()


class BaseFileLock(object):
    """Implements the base class of a file lock."""

    def __init__(self, lock_file, timeout=-1):
        """ """
        # The path to the lock file.
        self._lock_file = lock_file

        # The file descriptor for the *_lock_file* as it is returned by the os.open() function.
        # This file lock is only NOT None, if the object currently holds the lock.
        self._lock_file_fd = None

        # The default timeout value.
        self.timeout = timeout

        # We use this lock primarily for the lock counter.
        self._thread_lock = Lock()

        # The lock counter is used for implementing the nested locking mechanism. Whenever the lock is acquired, the
        # counter is increased and the lock is only released, when this value is 0 again.
        self._lock_counter = 0

    @property
    def lock_file(self):
        """The path to the lock file."""
        return self._lock_file

    @property
    def timeout(self):
        """
        You can set a default timeout for the filelock. It will be used as fallback value in the acquire method, if no
        timeout value (*None*) is given. If you want to disable the timeout, set it to a negative value. A timeout of
        0 means, that there is exactly one attempt to acquire the file lock.

        .. versionadded:: 2.0.0
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        """change the timeout parameter"""
        self._timeout = float(value)

    def _acquire(self):
        """If the file lock could be acquired, self._lock_file_fd holds the file descriptor of the lock file."""
        raise NotImplementedError

    def _release(self):
        """Releases the lock and sets self._lock_file_fd to None."""
        raise NotImplementedError

    @property
    def is_locked(self):
        """True, if the object holds the file lock.

        .. versionchanged:: 2.0.0

            This was previously a method and is now a property.
        """
        return self._lock_file_fd is not None

    def acquire(self, timeout=None, poll_intervall=0.05):
        """
        Acquires the file lock or fails with a :exc:`Timeout` error.

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            with lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            lock.acquire()
            try:
                pass
            finally:
                lock.release()

        :arg float timeout:
            The maximum time waited for the file lock.
            If ``timeout < 0``, there is no timeout and this method will
            block until the lock could be acquired.
            If ``timeout`` is None, the default :attr:`~timeout` is used.

        :arg float poll_intervall:
            We check once in *poll_intervall* seconds if we can acquire the
            file lock.

        :raises Timeout:
            if the lock could not be acquired in *timeout* seconds.

        .. versionchanged:: 2.0.0

            This method returns now a *proxy* object instead of *self*,
            so that it can be used in a with statement without side effects.
        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self.timeout

        # Increment the number right at the beginning. We can still undo it, if something fails.
        with self._thread_lock:
            self._lock_counter += 1

        lock_id = id(self)
        lock_filename = self._lock_file
        start_time = time.time()
        try:
            while True:
                with self._thread_lock:
                    if not self.is_locked:
                        _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
                        self._acquire()

                if self.is_locked:
                    _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    break
                elif 0 <= timeout < time.time() - start_time:
                    _LOGGER.debug("Timeout on acquiring lock %s on %s", lock_id, lock_filename)
                    raise Timeout(self._lock_file)
                else:
                    msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                    _LOGGER.debug(msg, lock_id, lock_filename, poll_intervall)
                    time.sleep(poll_intervall)
        except BaseException:  # Something did go wrong, so decrement the counter.
            with self._thread_lock:
                self._lock_counter = max(0, self._lock_counter - 1)
            raise
        return AcquireReturnProxy(lock=self)

    def release(self, force=False):
        """
        Releases the file lock.

        Please note, that the lock is only completely released, if the lock
        counter is 0.

        Also note, that the lock file itself is not automatically deleted.

        :arg bool force:
            If true, the lock counter is ignored and the lock is released in
            every case.
        """
        with self._thread_lock:

            if self.is_locked:
                self._lock_counter -= 1

                if self._lock_counter == 0 or force:
                    lock_id, lock_filename = id(self), self._lock_file

                    _LOGGER.debug("Attempting to release lock %s on %s", lock_id, lock_filename)
                    self._release()
                    self._lock_counter = 0
                    _LOGGER.debug("Lock %s released on %s", lock_id, lock_filename)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: U100
        self.release()

    def __del__(self):
        self.release(force=True)


__all__ = [
    "BaseFileLock",
    "AcquireReturnProxy",
]
