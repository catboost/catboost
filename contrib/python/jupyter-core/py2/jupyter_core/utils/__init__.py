import errno
import os

def ensure_dir_exists(path, mode=0o777):
    """ensure that a directory exists

    If it doesn't exist, try to create it, protecting against a race condition
    if another process is doing the same.

    The default permissions are determined by the current umask.
    """
    try:
        os.makedirs(path, mode=mode)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)
