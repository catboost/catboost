from __future__ import annotations

import os
import sys
from errno import ENOSYS
from typing import cast

from ._api import BaseFileLock

#: a flag to indicate if the fcntl API is available
has_fcntl = False
if sys.platform == "win32":  # pragma: win32 cover

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError

else:  # pragma: win32 no cover
    try:
        import fcntl
    except ImportError:
        pass
    else:
        has_fcntl = True

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            open_flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            fd = os.open(self._lock_file, open_flags, self._mode)
            try:
                os.fchmod(fd, self._mode)
            except PermissionError:
                pass  # This locked is not owned by this UID
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exception:
                os.close(fd)
                if exception.errno == ENOSYS:  # NotImplemented error
                    raise NotImplementedError("FileSystem does not appear to support flock; user SoftFileLock instead")
            else:
                self._lock_file_fd = fd

        def _release(self) -> None:
            # Do not remove the lockfile:
            #   https://github.com/tox-dev/py-filelock/issues/31
            #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
            fd = cast(int, self._lock_file_fd)
            self._lock_file_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = [
    "has_fcntl",
    "UnixFileLock",
]
