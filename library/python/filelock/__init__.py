import errno
import logging
import os
import sys

import library.python.windows

logger = logging.getLogger(__name__)


def set_close_on_exec(stream):
    if library.python.windows.on_win():
        library.python.windows.set_handle_information(stream, inherit=False)
    else:
        import fcntl
        fcntl.fcntl(stream, fcntl.F_SETFD, fcntl.FD_CLOEXEC)


class AbstractFileLock(object):

    def __init__(self, path):
        self.path = path

    def acquire(self, blocking=True):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()


class _NixFileLock(AbstractFileLock):

    def __init__(self, path):
        super(_NixFileLock, self).__init__(path)
        from fcntl import flock, LOCK_EX, LOCK_UN, LOCK_NB
        self._locker = lambda lock, blocking: flock(lock, LOCK_EX if blocking else LOCK_EX | LOCK_NB)
        self._unlocker = lambda lock: flock(lock, LOCK_UN)
        self._lock = open(self.path, 'a')
        set_close_on_exec(self._lock)

    def acquire(self, blocking=True):
        import errno
        try:
            self._locker(self._lock, blocking)
        except IOError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES) and not blocking:
                return False
            raise
        return True

    def release(self):
        self._unlocker(self._lock)

    def __del__(self):
        if hasattr(self, "_lock"):
            self._lock.close()


class _WinFileLock(AbstractFileLock):
    """
    Based on LockFile / UnlockFile from win32 API
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa365202(v=vs.85).aspx
    """

    _LOCKED_BYTES_NUM = 1

    def __init__(self, path):
        super(_WinFileLock, self).__init__(path)
        self._lock = None
        try:
            with open(path, 'w') as lock_file:
                lock_file.write(" " * self._LOCKED_BYTES_NUM)
        except IOError as e:
            if e.errno != errno.EACCES or not os.path.isfile(path):
                raise

    def acquire(self, blocking=True):
        self._lock = open(self.path)
        set_close_on_exec(self._lock)

        import time
        locked = False
        while not locked:
            locked = library.python.windows.lock_file(self._lock, 0, self._LOCKED_BYTES_NUM, raises=False)
            if locked:
                return True
            if blocking:
                time.sleep(.5)
            else:
                return False

    def release(self):
        if self._lock:
            library.python.windows.unlock_file(self._lock, 0, self._LOCKED_BYTES_NUM, raises=False)
            self._lock.close()
            self._lock = None


class FileLock(AbstractFileLock):

    def __init__(self, path):
        super(FileLock, self).__init__(path)

        if sys.platform.startswith('win'):
            self._lock = _WinFileLock(path)
        else:
            self._lock = _NixFileLock(path)

    def acquire(self, blocking=True):
        logger.debug('Acquiring filelock (blocking=%s): %s', blocking, self.path)
        return self._lock.acquire(blocking)

    def release(self):
        logger.debug('Ensuring filelock released: %s', self.path)
        return self._lock.release()
