import os
import stat
import sys

PermissionError = PermissionError if sys.version_info[0] == 3 else OSError


def raise_on_exist_ro_file(filename):
    try:
        file_stat = os.stat(filename)  # use stat to do exists + can write to check without race condition
    except OSError:
        return None  # swallow does not exist or other errors

    if file_stat.st_mtime != 0:  # if os.stat returns but modification is zero that's an invalid os.stat - ignore it
        if not (file_stat.st_mode & stat.S_IWUSR):
            raise PermissionError("Permission denied: {!r}".format(filename))


__all__ = [
    "raise_on_exist_ro_file",
    "PermissionError",
]
