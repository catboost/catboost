import collections
import errno
import logging
import os
import struct
import sys
import time

import library.python.windows

logger = logging.getLogger(__name__)

# python2 compat
os_O_CLOEXEC = getattr(os, 'O_CLOEXEC', 1 << 19)


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
        # nonbuffered random access rw mode
        self._lock = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT | os_O_CLOEXEC), 'r+b', 0)

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
        # nonbuffered random access rw mode
        self._lock = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_BINARY | os.O_NOINHERIT), 'r+b', 0)
        try:
            self._lock.write(b' ' * self._LOCKED_BYTES_NUM)
        except IOError as e:
            if e.errno != errno.EACCES or not os.path.isfile(path):
                raise

    def acquire(self, blocking=True):
        locked = False
        while not locked:
            locked = library.python.windows.lock_file(self._lock, 0, self._LOCKED_BYTES_NUM, raises=False)
            if locked:
                return True
            if blocking:
                time.sleep(0.5)
            else:
                return False

    def release(self):
        if self._lock:
            library.python.windows.unlock_file(self._lock, 0, self._LOCKED_BYTES_NUM, raises=False)

    def __del__(self):
        if getattr(self, '_lock', False):
            self._lock.close()


class FileLock(AbstractFileLock):

    def __init__(self, path):
        super(FileLock, self).__init__(path)

        if sys.platform.startswith('win'):
            self._lock = _WinFileLock(path)
        else:
            self._lock = _NixFileLock(path)

    def acquire(self, blocking=True):
        logger.debug('Acquiring %s (blocking=%s): %s', type(self).__name__, blocking, self.path)
        return self._lock.acquire(blocking)

    def release(self):
        logger.debug('Ensuring %s released: %s', type(self).__name__, self.path)
        return self._lock.release()


_LockInfo = collections.namedtuple('LockInfo', ['pid', 'time'])


class _PidLockMixin(object):
    _LockedBytes = 0
    _InfoFormat = 'QQ'
    _InfoFmtSize = struct.calcsize(_InfoFormat)

    def _register_lock(self):
        self._lock.seek(self._LockedBytes, os.SEEK_SET)
        self._lock.write(struct.pack(self._InfoFormat, os.getpid(), int(time.time())))

    @property
    def info(self):
        self._lock.seek(self._LockedBytes, os.SEEK_SET)
        try:
            data = struct.unpack(self._InfoFormat, self._lock.read(self._InfoFmtSize))
        except struct.error:
            data = 0, 0
        return _LockInfo(*data)


class _NixPidFileLock(_NixFileLock, _PidLockMixin):
    def acquire(self, blocking=True):
        if super(_NixPidFileLock, self).acquire(blocking):
            self._register_lock()
            return True
        return False


class _WinPidFileLock(_WinFileLock, _PidLockMixin):
    _LockedBytes = _WinFileLock._LOCKED_BYTES_NUM

    def acquire(self, blocking=True):
        if super(_WinPidFileLock, self).acquire(blocking):
            self._register_lock()
            return True
        return False


class PidFileLock(FileLock):

    def __init__(self, path):
        AbstractFileLock.__init__(self, path)

        if sys.platform.startswith('win'):
            self._lock = _WinPidFileLock(path)
        else:
            self._lock = _NixPidFileLock(path)

    @property
    def info(self):
        return self._lock.info
