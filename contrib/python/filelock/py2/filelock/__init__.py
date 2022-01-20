"""
A platform independent file lock that supports the with-statement.

.. autodata:: filelock.__version__
   :no-value:

"""
import sys
import warnings

from ._api import AcquireReturnProxy, BaseFileLock
from ._error import Timeout
from ._soft import SoftFileLock
from ._unix import UnixFileLock, has_fcntl
from ._windows import WindowsFileLock
from .version import version

#: version of the project as a string
__version__ = version


if sys.platform == "win32":
    _FileLock = WindowsFileLock
elif has_fcntl:
    _FileLock = UnixFileLock
else:
    _FileLock = SoftFileLock
    if warnings is not None:
        warnings.warn("only soft file lock is available")

#: Alias for the lock, which should be used for the current platform. On Windows, this is an alias for
# :class:`WindowsFileLock`, on Unix for :class:`UnixFileLock` and otherwise for :class:`SoftFileLock`.
FileLock = _FileLock


__all__ = [
    "__version__",
    "FileLock",
    "SoftFileLock",
    "WindowsFileLock",
    "UnixFileLock",
    "BaseFileLock",
    "Timeout",
    "AcquireReturnProxy",
]
