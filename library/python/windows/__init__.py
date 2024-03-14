# coding: utf-8

import os
import stat
import sys
import shutil
import logging

from six import reraise

import library.python.func
import library.python.strings

logger = logging.getLogger(__name__)


ERRORS = {
    'SUCCESS': 0,
    'PATH_NOT_FOUND': 3,
    'ACCESS_DENIED': 5,
    'SHARING_VIOLATION': 32,
    'INSUFFICIENT_BUFFER': 122,
    'DIR_NOT_EMPTY': 145,
}

RETRIABLE_FILE_ERRORS = (ERRORS['ACCESS_DENIED'], ERRORS['SHARING_VIOLATION'])
RETRIABLE_DIR_ERRORS = (ERRORS['ACCESS_DENIED'], ERRORS['DIR_NOT_EMPTY'], ERRORS['SHARING_VIOLATION'])


@library.python.func.lazy
def on_win():
    """Check if code run on Windows"""
    return os.name == 'nt'


class NotOnWindowsError(RuntimeError):
    def __init__(self, message):
        super(NotOnWindowsError, self).__init__(message)


class DisabledOnWindowsError(RuntimeError):
    def __init__(self, message):
        super(DisabledOnWindowsError, self).__init__(message)


class NoCTypesError(RuntimeError):
    def __init__(self, message):
        super(NoCTypesError, self).__init__(message)


def win_only(f):
    """Decorator for Windows-only functions"""

    def f_wrapped(*args, **kwargs):
        if not on_win():
            raise NotOnWindowsError('Windows-only function is called, but platform is not Windows')
        return f(*args, **kwargs)

    return f_wrapped


def win_disabled(f):
    """Decorator for functions disabled on Windows"""
    def f_wrapped(*args, **kwargs):
        if on_win():
            run_disabled()
        return f(*args, **kwargs)

    return f_wrapped


def errorfix(f):
    if not on_win():
        return f

    def f_wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except WindowsError:
            tp, value, tb = sys.exc_info()
            fix_error(value)
            reraise(tp, value, tb)

    return f_wrapped


def diehard(winerrors, tries=100, delay=1):
    """
    Decorator for diehard wrapper

    On Windows platform retries to run function while specific WindowsError is thrown

    On non-Windows platforms fallbacks to function itself
    """
    def wrap(f):
        if not on_win():
            return f

        return lambda *args, **kwargs: run_diehard(f, winerrors, tries, delay, *args, **kwargs)

    return wrap


def win_path_fix(path):
    """Fix slashes in paths on windows"""
    return path if sys.platform != 'win32' else path.replace('\\', '/')


if on_win():
    import msvcrt
    import time

    import library.python.strings

    _has_ctypes = True
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        _has_ctypes = False

    _INVALID_HANDLE_VALUE = -1

    _MOVEFILE_REPLACE_EXISTING = 0x1
    _MOVEFILE_WRITE_THROUGH = 0x8

    _SEM_FAILCRITICALERRORS = 0x1
    _SEM_NOGPFAULTERRORBOX = 0x2
    _SEM_NOALIGNMENTFAULTEXCEPT = 0x4
    _SEM_NOOPENFILEERRORBOX = 0x8

    _SYMBOLIC_LINK_FLAG_DIRECTORY = 0x1

    _CREATE_NO_WINDOW = 0x8000000

    _ATOMIC_RENAME_FILE_TRANSACTION_DEFAULT_TIMEOUT = 1000

    _HANDLE_FLAG_INHERIT = 0x1

    @win_only
    def require_ctypes(f):
        def f_wrapped(*args, **kwargs):
            if not _has_ctypes:
                raise NoCTypesError('No ctypes found')
            return f(*args, **kwargs)

        return f_wrapped

    # Run function in diehard mode (see diehard decorator commentary)
    @win_only
    def run_diehard(f, winerrors, tries, delay, *args, **kwargs):
        if isinstance(winerrors, int):
            winerrors = (winerrors,)

        ei = None
        for t in range(tries):
            if t:
                logger.debug('Diehard [errs %s]: try #%d in %s', ','.join(str(x) for x in winerrors), t, f)
            try:
                return f(*args, **kwargs)
            except WindowsError as e:
                if e.winerror not in winerrors:
                    raise
                ei = sys.exc_info()
                time.sleep(delay)
        reraise(ei[0], ei[1], ei[2])

    @win_only
    def run_disabled(*args, **kwargs):
        """Placeholder for disabled functions"""
        raise DisabledOnWindowsError('Function called is disabled on Windows')

    class CustomWinError(WindowsError):
        def __init__(self, winerror, message='', filename=None):
            super(CustomWinError, self).__init__(winerror, message)
            self.message = message
            self.strerror = self.message if self.message else format_error(self.windows_error)
            self.filename = filename
            self.utf8 = True

    @win_only
    def unicode_path(path):
        return library.python.strings.to_unicode(path, library.python.strings.fs_encoding())

    @win_only
    @require_ctypes
    def format_error(error):
        if isinstance(error, WindowsError):
            error = error.winerror
        if not isinstance(error, int):
            return 'Unknown'
        return ctypes.FormatError(error)

    @win_only
    def fix_error(windows_error):
        if not windows_error.strerror:
            windows_error.strerror = format_error(windows_error)
        transcode_error(windows_error)

    @win_only
    def transcode_error(windows_error, to_enc='utf-8'):
        from_enc = 'utf-8' if getattr(windows_error, 'utf8', False) else library.python.strings.guess_default_encoding()
        if from_enc != to_enc:
            windows_error.strerror = library.python.strings.to_str(
                windows_error.strerror, to_enc=to_enc, from_enc=from_enc
            )
        setattr(windows_error, 'utf8', to_enc == 'utf-8')

    class Transaction(object):
        def __init__(self, timeout=None, description=''):
            self.timeout = timeout
            self.description = description

        @require_ctypes
        def __enter__(self):
            self._handle = ctypes.windll.ktmw32.CreateTransaction(None, 0, 0, 0, 0, self.timeout, self.description)
            if self._handle == _INVALID_HANDLE_VALUE:
                raise ctypes.WinError()
            return self._handle

        @require_ctypes
        def __exit__(self, t, v, tb):
            try:
                if not ctypes.windll.ktmw32.CommitTransaction(self._handle):
                    raise ctypes.WinError()
            finally:
                ctypes.windll.kernel32.CloseHandle(self._handle)

    @win_only
    def file_handle(f):
        return msvcrt.get_osfhandle(f.fileno())

    # https://www.python.org/dev/peps/pep-0446/
    # http://mihalop.blogspot.ru/2014/05/python-subprocess-and-file-descriptors.html
    @require_ctypes
    @win_only
    def open_file(*args, **kwargs):
        f = open(*args, **kwargs)
        ctypes.windll.kernel32.SetHandleInformation(file_handle(f), _HANDLE_FLAG_INHERIT, 0)
        return f

    @win_only
    @require_ctypes
    def replace_file(src, dst):
        if not ctypes.windll.kernel32.MoveFileExW(
            unicode_path(src), unicode_path(dst), _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH
        ):
            raise ctypes.WinError()

    @win_only
    @require_ctypes
    def replace_file_across_devices(src, dst):
        with Transaction(
            timeout=_ATOMIC_RENAME_FILE_TRANSACTION_DEFAULT_TIMEOUT,
            description='ya library.python.windows replace_file_across_devices',
        ) as transaction:
            if not ctypes.windll.kernel32.MoveFileTransactedW(
                unicode_path(src),
                unicode_path(dst),
                None,
                None,
                _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH,
                transaction,
            ):
                raise ctypes.WinError()

    @win_only
    @require_ctypes
    def hardlink(src, lnk):
        if not ctypes.windll.kernel32.CreateHardLinkW(unicode_path(lnk), unicode_path(src), None):
            raise ctypes.WinError()

    # Requires SE_CREATE_SYMBOLIC_LINK_NAME privilege
    @win_only
    @win_disabled
    @require_ctypes
    def symlink_file(src, lnk):
        if not ctypes.windll.kernel32.CreateSymbolicLinkW(unicode_path(lnk), unicode_path(src), 0):
            raise ctypes.WinError()

    # Requires SE_CREATE_SYMBOLIC_LINK_NAME privilege
    @win_only
    @win_disabled
    @require_ctypes
    def symlink_dir(src, lnk):
        if not ctypes.windll.kernel32.CreateSymbolicLinkW(
            unicode_path(lnk), unicode_path(src), _SYMBOLIC_LINK_FLAG_DIRECTORY
        ):
            raise ctypes.WinError()

    @win_only
    @require_ctypes
    def lock_file(f, offset, length, raises=True):
        locked = ctypes.windll.kernel32.LockFile(
            file_handle(f), _low_dword(offset), _high_dword(offset), _low_dword(length), _high_dword(length)
        )
        if not raises:
            return bool(locked)
        if not locked:
            raise ctypes.WinError()

    @win_only
    @require_ctypes
    def unlock_file(f, offset, length, raises=True):
        unlocked = ctypes.windll.kernel32.UnlockFile(
            file_handle(f), _low_dword(offset), _high_dword(offset), _low_dword(length), _high_dword(length)
        )
        if not raises:
            return bool(unlocked)
        if not unlocked:
            raise ctypes.WinError()

    @win_only
    @require_ctypes
    def set_error_mode(mode):
        return ctypes.windll.kernel32.SetErrorMode(mode)

    @win_only
    def rmtree(path):
        def error_handler(func, handling_path, execinfo):
            e = execinfo[1]
            if e.winerror == ERRORS['PATH_NOT_FOUND']:
                handling_path = "\\\\?\\" + handling_path  # handle path over 256 symbols
                if os.path.exists(path):
                    return func(handling_path)
            if e.winerror == ERRORS['ACCESS_DENIED']:
                try:
                    # removing of r/w directory with read-only files in it yields ACCESS_DENIED
                    # which is not an insuperable obstacle https://bugs.python.org/issue19643
                    os.chmod(handling_path, stat.S_IWRITE)
                except OSError:
                    pass
                else:
                    # propagate true last error if this attempt fails
                    return func(handling_path)
            raise e

        shutil.rmtree(path, onerror=error_handler)

    # Don't display the Windows GPF dialog if the invoked program dies.
    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
    @win_only
    def disable_error_dialogs():
        set_error_mode(_SEM_NOGPFAULTERRORBOX | _SEM_FAILCRITICALERRORS)

    @win_only
    def default_process_creation_flags():
        return 0

    @require_ctypes
    def _low_dword(x):
        return ctypes.c_ulong(x & ((1 << 32) - 1))

    @require_ctypes
    def _high_dword(x):
        return ctypes.c_ulong((x >> 32) & ((1 << 32) - 1))

    @win_only
    @require_ctypes
    def get_current_process():
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not handle:
            raise ctypes.WinError()
        return wintypes.HANDLE(handle)

    @win_only
    @require_ctypes
    def get_process_handle_count(proc_handle):
        assert isinstance(proc_handle, wintypes.HANDLE)

        GetProcessHandleCount = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.POINTER(wintypes.DWORD))(
            ("GetProcessHandleCount", ctypes.windll.kernel32)
        )
        hndcnt = wintypes.DWORD()
        if not GetProcessHandleCount(proc_handle, ctypes.byref(hndcnt)):
            raise ctypes.WinError()
        return hndcnt.value

    @win_only
    @require_ctypes
    def set_handle_information(file, inherit=None, protect_from_close=None):
        for flag, value in [(inherit, 1), (protect_from_close, 2)]:
            if flag is not None:
                assert isinstance(flag, bool)
                if not ctypes.windll.kernel32.SetHandleInformation(
                    file_handle(file), _low_dword(value), _low_dword(int(flag))
                ):
                    raise ctypes.WinError()

    @win_only
    @require_ctypes
    def get_windows_directory():
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        size = ctypes.windll.kernel32.GetWindowsDirectoryW(buf, ctypes.wintypes.MAX_PATH)
        if not size:
            raise ctypes.WinError()
        if size > ctypes.wintypes.MAX_PATH - 1:
            raise CustomWinError(ERRORS['INSUFFICIENT_BUFFER'])
        return ctypes.wstring_at(buf, size)
