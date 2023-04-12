from __future__ import annotations

import os
import sys
from errno import EACCES, EEXIST

from ._api import BaseFileLock
from ._util import raise_on_exist_ro_file


class SoftFileLock(BaseFileLock):
    """Simply watches the existence of the lock file."""

    def _acquire(self) -> None:
        raise_on_exist_ro_file(self._lock_file)
        # first check for exists and read-only mode as the open will mask this case as EEXIST
        flags = (
            os.O_WRONLY  # open for writing only
            | os.O_CREAT
            | os.O_EXCL  # together with above raise EEXIST if the file specified by filename exists
            | os.O_TRUNC  # truncate the file to zero byte
        )
        try:
            file_handler = os.open(self._lock_file, flags, self._mode)
        except OSError as exception:  # re-raise unless expected exception
            if not (
                exception.errno == EEXIST  # lock already exist
                or (exception.errno == EACCES and sys.platform == "win32")  # has no access to this lock
            ):  # pragma: win32 no cover
                raise
        else:
            self._lock_file_fd = file_handler

    def _release(self) -> None:
        os.close(self._lock_file_fd)  # type: ignore # the lock file is definitely not None
        self._lock_file_fd = None
        try:
            os.remove(self._lock_file)
        except OSError:  # the file is already deleted and that's what we want
            pass


__all__ = [
    "SoftFileLock",
]
