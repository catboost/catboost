import sys

if sys.version[0] == 3:
    TimeoutError = TimeoutError
else:
    TimeoutError = OSError


class Timeout(TimeoutError):
    """Raised when the lock could not be acquired in *timeout* seconds."""

    def __init__(self, lock_file):
        #: The path of the file lock.
        self.lock_file = lock_file

    def __str__(self):
        return "The file lock '{}' could not be acquired.".format(self.lock_file)


__all__ = [
    "Timeout",
]
