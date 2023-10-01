import contextlib
import os
import typing

from . import constants, exceptions

# Alias for readability. Due to import recursion issues we cannot do:
# from .constants import LockFlags
LockFlags = constants.LockFlags


if os.name == 'nt':  # pragma: no cover
    import msvcrt

    import pywintypes
    import win32con
    import win32file
    import winerror

    __overlapped = pywintypes.OVERLAPPED()

    def lock(file_: typing.Union[typing.IO, int], flags: LockFlags):
        # Windows locking does not support locking through `fh.fileno()` so
        # we cast it to make mypy and pyright happy
        file_ = typing.cast(typing.IO, file_)

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

        os_fh = msvcrt.get_osfhandle(file_.fileno())  # type: ignore
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

    def unlock(file_: typing.IO):
        try:
            savepos = file_.tell()
            if savepos:
                file_.seek(0)

            os_fh = msvcrt.get_osfhandle(file_.fileno())  # type: ignore
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
    import fcntl

    def lock(file_: typing.Union[typing.IO, int], flags: LockFlags):
        locking_exceptions = (IOError,)
        with contextlib.suppress(NameError):
            locking_exceptions += (BlockingIOError,)  # type: ignore
        # Locking with NON_BLOCKING without EXCLUSIVE or SHARED enabled results
        # in an error
        if (flags & LockFlags.NON_BLOCKING) and not flags & (
            LockFlags.SHARED | LockFlags.EXCLUSIVE
        ):
            raise RuntimeError(
                'When locking in non-blocking mode the SHARED '
                'or EXCLUSIVE flag must be specified as well',
            )

        try:
            fcntl.flock(file_, flags)
        except locking_exceptions as exc_value:
            # The exception code varies on different systems so we'll catch
            # every IO error
            raise exceptions.LockException(exc_value, fh=file_) from exc_value

    def unlock(file_: typing.IO):
        fcntl.flock(file_.fileno(), LockFlags.UNBLOCK)

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')
