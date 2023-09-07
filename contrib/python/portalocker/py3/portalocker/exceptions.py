import typing


class BaseLockException(Exception):
    # Error codes:
    LOCK_FAILED = 1

    def __init__(
        self,
        *args: typing.Any,
        fh: typing.Optional[typing.IO] = None,
        **kwargs: typing.Any,
    ) -> None:
        self.fh = fh
        Exception.__init__(self, *args)


class LockException(BaseLockException):
    pass


class AlreadyLocked(LockException):
    pass


class FileToLarge(LockException):
    pass
