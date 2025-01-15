from __future__ import annotations

import os
import typing

from . import constants, exceptions, types

# Alias for readability. Due to import recursion issues we cannot do:
# from .constants import LockFlags
LockFlags = constants.LockFlags


class HasFileno(typing.Protocol):
    def fileno(self) -> int: ...


LOCKER: typing.Callable[[int | HasFileno, int], typing.Any] | None = None

if os.name == 'nt':  # pragma: no cover
    import msvcrt

    import pywintypes
    import win32con
    import win32file
    import winerror

    __overlapped = pywintypes.OVERLAPPED()

    def lock(file_: types.IO | int, flags: LockFlags) -> None:
        # Windows locking does not support locking through `fh.fileno()` so
        # we cast it to make mypy and pyright happy
        file_ = typing.cast(types.IO, file_)

        mode = 0
        if flags & LockFlags.NON_BLOCKING:
            mode |= win32con.LOCKFILE_FAIL_IMMEDIATELY

        if flags & LockFlags.EXCLUSIVE:
            mode |= win32con.LOCKFILE_EXCLUSIVE_LOCK

        # Save the old position so we can go back to that position but
        # still lock from the beginning of the file
        savepos = file_.tell()
        if savepos:
            file_.seek(0)

        os_fh = msvcrt.get_osfhandle(file_.fileno())  # type: ignore[attr-defined]
        try:
            win32file.LockFileEx(os_fh, mode, 0, -0x10000, __overlapped)
        except pywintypes.error as exc_value:
            # error: (33, 'LockFileEx', 'The process cannot access the file
            # because another process has locked a portion of the file.')
            if exc_value.winerror == winerror.ERROR_LOCK_VIOLATION:
                raise exceptions.AlreadyLocked(
                    exceptions.LockException.LOCK_FAILED,
                    exc_value.strerror,
                    fh=file_,
                ) from exc_value
            else:
                # Q:  Are there exceptions/codes we should be dealing with
                # here?
                raise
        finally:
            if savepos:
                file_.seek(savepos)

    def unlock(file_: types.IO) -> None:
        try:
            savepos = file_.tell()
            if savepos:
                file_.seek(0)

            os_fh = msvcrt.get_osfhandle(file_.fileno())  # type: ignore[attr-defined]
            try:
                win32file.UnlockFileEx(
                    os_fh,
                    0,
                    -0x10000,
                    __overlapped,
                )
            except pywintypes.error as exc:
                if exc.winerror != winerror.ERROR_NOT_LOCKED:
                    # Q:  Are there exceptions/codes we should be
                    # dealing with here?
                    raise
            finally:
                if savepos:
                    file_.seek(savepos)
        except OSError as exc:
            raise exceptions.LockException(
                exceptions.LockException.LOCK_FAILED,
                exc.strerror,
                fh=file_,
            ) from exc

elif os.name == 'posix':  # pragma: no cover
    import errno
    import fcntl

    # The locking implementation.
    # Expected values are either fcntl.flock() or fcntl.lockf(),
    # but any callable that matches the syntax will be accepted.
    LOCKER = fcntl.flock  # pyright: ignore[reportConstantRedefinition]

    def lock(file: int | types.IO, flags: LockFlags) -> None:  # type: ignore[misc]
        assert LOCKER is not None, 'We need a locking function in `LOCKER` '
        # Locking with NON_BLOCKING without EXCLUSIVE or SHARED enabled
        # results in an error
        if (flags & LockFlags.NON_BLOCKING) and not flags & (
            LockFlags.SHARED | LockFlags.EXCLUSIVE
        ):
            raise RuntimeError(
                'When locking in non-blocking mode the SHARED '
                'or EXCLUSIVE flag must be specified as well',
            )

        try:
            LOCKER(file, flags)
        except OSError as exc_value:
            # Python can use one of several different exception classes to
            # represent timeout (most likely is BlockingIOError and IOError),
            # but these errors may also represent other failures. On some
            # systems, `IOError is OSError` which means checking for either
            # IOError or OSError can mask other errors.
            # The safest check is to catch OSError (from which the others
            # inherit) and check the errno (which should be EACCESS or EAGAIN
            # according to the spec).
            if exc_value.errno in (errno.EACCES, errno.EAGAIN):
                # A timeout exception, wrap this so the outer code knows to try
                # again (if it wants to).
                raise exceptions.AlreadyLocked(
                    exc_value,
                    fh=file,
                ) from exc_value
            else:
                # Something else went wrong; don't wrap this so we stop
                # immediately.
                raise exceptions.LockException(
                    exc_value,
                    fh=file,
                ) from exc_value
        except EOFError as exc_value:
            # On NFS filesystems, flock can raise an EOFError
            raise exceptions.LockException(
                exc_value,
                fh=file,
            ) from exc_value

    def unlock(file: types.IO) -> None:  # type: ignore[misc]
        assert LOCKER is not None, 'We need a locking function in `LOCKER` '
        LOCKER(file.fileno(), LockFlags.UNBLOCK)

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')
