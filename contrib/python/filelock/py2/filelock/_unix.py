import os

from ._api import BaseFileLock

try:
    import fcntl
except ImportError:
    fcntl = None

#: a flag to indicate if the fcntl API is available
has_fcntl = fcntl is not None


class UnixFileLock(BaseFileLock):
    """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

    def _acquire(self):
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        fd = os.open(self._lock_file, open_mode)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):  # noqa: B014 # IOError is not OSError on python 2
            os.close(fd)
        else:
            self._lock_file_fd = fd

    def _release(self):
        # Do not remove the lockfile:
        #   https://github.com/tox-dev/py-filelock/issues/31
        #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
        fd = self._lock_file_fd
        self._lock_file_fd = None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


__all__ = [
    "has_fcntl",
    "UnixFileLock",
]
