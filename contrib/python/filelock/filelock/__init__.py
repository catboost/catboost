"""
A platform independent file lock that supports the with-statement.
"""
import sys
import warnings

from ._api import AcquireReturnProxy, BaseFileLock
from ._error import Timeout
from ._soft import SoftFileLock
from ._unix import UnixFileLock, has_fcntl
from ._windows import WindowsFileLock
from .version import version as __version__

#: Alias for the lock, which should be used for the current platform. On Windows, this is an alias for
# :class:`WindowsFileLock`, on Unix for :class:`UnixFileLock` and otherwise for :class:`SoftFileLock`.
FileLock = None

if sys.platform == "win32":
    FileLock = WindowsFileLock
elif has_fcntl:
    FileLock = UnixFileLock
else:
    FileLock = SoftFileLock

    if warnings is not None:
        warnings.warn("only soft file lock is available")

__all__ = [
    "Timeout",
    "BaseFileLock",
    "WindowsFileLock",
    "UnixFileLock",
    "SoftFileLock",
    "FileLock",
    "AcquireReturnProxy",
    "__version__",
]
