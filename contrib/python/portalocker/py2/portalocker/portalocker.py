import os
import sys
from . import exceptions
from . import constants


if os.name == 'nt':  # pragma: no cover
    import win32con
    import win32file
    import pywintypes
    import winerror
    import msvcrt
    __overlapped = pywintypes.OVERLAPPED()

    if sys.version_info.major == 2:
        lock_length = -1
    else:
        lock_length = int(2**31 - 1)

    def lock(file_, flags):
        if flags & constants.LOCK_SH:
            if sys.version_info.major == 2:
                if flags & constants.LOCK_NB:
                    mode = win32con.LOCKFILE_FAIL_IMMEDIATELY
                else:
                    mode = 0

            else:
                if flags & constants.LOCK_NB:
                    mode = msvcrt.LK_NBRLCK
                else:
                    mode = msvcrt.LK_RLCK

            # is there any reason not to reuse the following structure?
            hfile = win32file._get_osfhandle(file_.fileno())
            try:
                win32file.LockFileEx(hfile, mode, 0, -0x10000, __overlapped)
            except pywintypes.error as exc_value:
                # error: (33, 'LockFileEx', 'The process cannot access the file
                # because another process has locked a portion of the file.')
                if exc_value.winerror == winerror.ERROR_LOCK_VIOLATION:
                    raise exceptions.LockException(
                        exceptions.LockException.LOCK_FAILED,
                        exc_value.strerror,
                        fh=file_)
                else:
                    # Q:  Are there exceptions/codes we should be dealing with
                    # here?
                    raise
        else:
            mode = win32con.LOCKFILE_EXCLUSIVE_LOCK
            if flags & constants.LOCK_NB:
                mode |= win32con.LOCKFILE_FAIL_IMMEDIATELY

            if flags & constants.LOCK_NB:
                mode = msvcrt.LK_NBLCK
            else:
                mode = msvcrt.LK_LOCK

            # windows locks byte ranges, so make sure to lock from file start
            try:
                savepos = file_.tell()
                if savepos:
                    # [ ] test exclusive lock fails on seek here
                    # [ ] test if shared lock passes this point
                    file_.seek(0)
                    # [x] check if 0 param locks entire file (not documented in
                    #     Python)
                    # [x] fails with "IOError: [Errno 13] Permission denied",
                    #     but -1 seems to do the trick

                try:
                    msvcrt.locking(file_.fileno(), mode, lock_length)
                except IOError as exc_value:
                    # [ ] be more specific here
                    raise exceptions.LockException(
                        exceptions.LockException.LOCK_FAILED,
                        exc_value.strerror,
                        fh=file_)
                finally:
                    if savepos:
                        file_.seek(savepos)
            except IOError as exc_value:
                raise exceptions.LockException(
                    exceptions.LockException.LOCK_FAILED, exc_value.strerror,
                    fh=file_)

    def unlock(file_):
        try:
            savepos = file_.tell()
            if savepos:
                file_.seek(0)

            try:
                msvcrt.locking(file_.fileno(), constants.LOCK_UN, lock_length)
            except IOError as exc_value:
                if exc_value.strerror == 'Permission denied':
                    hfile = win32file._get_osfhandle(file_.fileno())
                    try:
                        win32file.UnlockFileEx(
                            hfile, 0, -0x10000, __overlapped)
                    except pywintypes.error as exc_value:
                        if exc_value.winerror == winerror.ERROR_NOT_LOCKED:
                            # error: (158, 'UnlockFileEx',
                            #         'The segment is already unlocked.')
                            # To match the 'posix' implementation, silently
                            # ignore this error
                            pass
                        else:
                            # Q:  Are there exceptions/codes we should be
                            # dealing with here?
                            raise
                else:
                    raise exceptions.LockException(
                        exceptions.LockException.LOCK_FAILED,
                        exc_value.strerror,
                        fh=file_)
            finally:
                if savepos:
                    file_.seek(savepos)
        except IOError as exc_value:
            raise exceptions.LockException(
                exceptions.LockException.LOCK_FAILED, exc_value.strerror,
                fh=file_)

elif os.name == 'posix':  # pragma: no cover
    import fcntl

    def lock(file_, flags):
        locking_exceptions = IOError,
        try:  # pragma: no cover
            locking_exceptions += BlockingIOError,
        except NameError:  # pragma: no cover
            pass

        try:
            fcntl.flock(file_.fileno(), flags)
        except locking_exceptions as exc_value:
            # The exception code varies on different systems so we'll catch
            # every IO error
            raise exceptions.LockException(exc_value, fh=file_)

    def unlock(file_):
        fcntl.flock(file_.fileno(), constants.LOCK_UN)

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')

