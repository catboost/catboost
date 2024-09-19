class BaseLockException(Exception):
    # Error codes:
    LOCK_FAILED = 1

    def __init__(self, *args, **kwargs):
        self.fh = kwargs.pop('fh', None)
        Exception.__init__(self, *args, **kwargs)


class LockException(BaseLockException):
    pass


class AlreadyLocked(BaseLockException):
    pass


class FileToLarge(BaseLockException):
    pass
