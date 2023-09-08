from . import __about__
from . import constants
from . import exceptions
from . import portalocker
from . import utils

#: The package name on Pypi
__package_name__ = __about__.__package_name__
#: Current author and maintainer, view the git history for the previous ones
__author__ = __about__.__author__
#: Current author's email address
__email__ = __about__.__email__
#: Version number
__version__ = '1.7.1'
#: Package description for Pypi
__description__ = __about__.__description__
#: Package homepage
__url__ = __about__.__url__


#: Exception thrown when the file is already locked by someone else
AlreadyLocked = exceptions.AlreadyLocked
#: Exception thrown if an error occurred during locking
LockException = exceptions.LockException


#: Lock a file. Note that this is an advisory lock on Linux/Unix systems
lock = portalocker.lock
#: Unlock a file
unlock = portalocker.unlock

#: Place an exclusive lock.
#: Only one process may hold an exclusive lock for a given file at a given
#: time.
LOCK_EX = constants.LOCK_EX

#: Place a shared lock.
#: More than one process may hold a shared lock for a given file at a given
#: time.
LOCK_SH = constants.LOCK_SH

#: Acquire the lock in a non-blocking fashion.
LOCK_NB = constants.LOCK_NB

#: Remove an existing lock held by this process.
LOCK_UN = constants.LOCK_UN

#: Locking utility class to automatically handle opening with timeouts and
#: context wrappers
Lock = utils.Lock
RLock = utils.RLock
TemporaryFileLock = utils.TemporaryFileLock
open_atomic = utils.open_atomic

__all__ = [
    'lock',
    'unlock',
    'LOCK_EX',
    'LOCK_SH',
    'LOCK_NB',
    'LOCK_UN',
    'LockException',
    'Lock',
    'AlreadyLocked',
    'open_atomic',
]

