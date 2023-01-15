import os
import stat

try:
    from pathlib import Path
except ImportError:
    try:
        # Python 2 backport
        from pathlib2 import Path
    except ImportError:
        class Path(object):
            """Dummy for isinstance checks"""
            pass


__all__ = ['assert_path_exists', 'assert_not_path_exists',
           'assert_isfile', 'assert_not_isfile',
           'assert_isdir', 'assert_not_isdir',
           'assert_islink', 'assert_not_islink',
           'assert_ispipe', 'assert_not_ispipe',
           'assert_issocket', 'assert_not_issocket',
          ]

def _strpath(p):
    if isinstance(p, Path):
        return str(p)
    return p

def _stat_for_assert(path, follow_symlinks=True, msg=None):
    stat = os.stat if follow_symlinks else os.lstat
    try:
        return stat(path)
    except OSError:
        if msg is None:
            msg = "Path does not exist, or can't be stat-ed: %r" % path
        raise AssertionError(msg)

def assert_path_exists(path, msg=None):
    """Assert that something exists at the given path.
    """
    _stat_for_assert(_strpath(path), True, msg)

def assert_not_path_exists(path, msg=None):
    """Assert that nothing exists at the given path.
    """
    path = _strpath(path)
    if os.path.exists(path):
        if msg is None:
            msg = "Path exists: %r" % path
        raise AssertionError(msg)

def assert_isfile(path, follow_symlinks=True, msg=None):
    """Assert that path exists and is a regular file.
    
    With follow_symlinks=True, the default, this will pass if path is a symlink
    to a regular file. With follow_symlinks=False, it will fail in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if not stat.S_ISREG(st.st_mode):
        if msg is None:
            msg = "Path exists, but is not a regular file: %r" % path
        raise AssertionError(msg)

def assert_not_isfile(path, follow_symlinks=True, msg=None):
    """Assert that path exists but is not a regular file.
    
    With follow_symlinks=True, the default, this will fail if path is a symlink
    to a regular file. With follow_symlinks=False, it will pass in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if stat.S_ISREG(st.st_mode):
        if msg is None:
            msg = "Path is a regular file: %r" % path
        raise AssertionError(msg)

def assert_isdir(path, follow_symlinks=True, msg=None):
    """Assert that path exists and is a directory.
    
    With follow_symlinks=True, the default, this will pass if path is a symlink
    to a directory. With follow_symlinks=False, it will fail in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if not stat.S_ISDIR(st.st_mode):
        if msg is None:
            msg = "Path exists, but is not a directory: %r" % path
        raise AssertionError(msg)

def assert_not_isdir(path, follow_symlinks=True, msg=None):
    """Assert that path exists but is not a directory.
    
    With follow_symlinks=True, the default, this will fail if path is a symlink
    to a directory. With follow_symlinks=False, it will pass in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if stat.S_ISDIR(st.st_mode):
        if msg is None:
            msg = "Path is a directory: %r" % path
        raise AssertionError(msg)

_link_target_msg = """Symlink target of:
  {path}
Expected:
  {expected}
Actual:
  {actual}
"""

def assert_islink(path, to=None, msg=None):
    """Assert that path exists and is a symlink.
    
    If to is specified, also check that it is the target of the symlink.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, False, msg)
    if not stat.S_ISLNK(st.st_mode):
        if msg is None:
            msg = "Path exists, but is not a symlink: %r" % path
        raise AssertionError(msg)
    
    if to is not None:
        to = _strpath(to)
        target = os.readlink(path)
        # TODO: Normalise the target to an absolute path?
        if target != to:
            if msg is None:
                msg = _link_target_msg.format(path=path, expected=to, actual=target)
            raise AssertionError(msg)

def assert_not_islink(path, msg=None):
    """Assert that path exists but is not a symlink.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, False, msg)
    if stat.S_ISLNK(st.st_mode):
        if msg is None:
            msg = "Path is a symlink: %r" % path
        raise AssertionError(msg)

def assert_ispipe(path, follow_symlinks=True, msg=None):
    """Assert that path exists and is a named pipe (FIFO).

    With follow_symlinks=True, the default, this will pass if path is a symlink
    to a named pipe. With follow_symlinks=False, it will fail in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if not stat.S_ISFIFO(st.st_mode):
        if msg is None:
            msg = "Path exists, but is not a named pipe: %r" % path
        raise AssertionError(msg)

def assert_not_ispipe(path, follow_symlinks=True, msg=None):
    """Assert that path exists but is not a named pipe (FIFO).

    With follow_symlinks=True, the default, this will fail if path is a symlink
    to a named pipe. With follow_symlinks=False, it will pass in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if stat.S_ISFIFO(st.st_mode):
        if msg is None:
            msg = "Path is a named pipe: %r" % path
        raise AssertionError(msg)

def assert_issocket(path, follow_symlinks=True, msg=None):
    """Assert that path exists and is a Unix domain socket.

    With follow_symlinks=True, the default, this will pass if path is a symlink
    to a Unix domain socket. With follow_symlinks=False, it will fail in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if not stat.S_ISSOCK(st.st_mode):
        if msg is None:
            msg = "Path exists, but is not a socket: %r" % path
        raise AssertionError(msg)

def assert_not_issocket(path, follow_symlinks=True, msg=None):
    """Assert that path exists but is not a Unix domain socket.

    With follow_symlinks=True, the default, this will fail if path is a symlink
    to a Unix domain socket. With follow_symlinks=False, it will pass in that case.
    """
    path = _strpath(path)
    st = _stat_for_assert(path, follow_symlinks, msg)
    if stat.S_ISSOCK(st.st_mode):
        if msg is None:
            msg = "Path is a socket: %r" % path
        raise AssertionError(msg)
