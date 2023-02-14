import os

import library.python.windows
import six


def normpath(path):
    if library.python.windows.on_win():
        return os.path.normcase(path).replace('\\', '/')
    return path


def abspath(path, base=None, expand_user=False):
    if expand_user:
        path = os.path.expanduser(path)
    if not os.path.isabs(path):
        if base is None:
            base = os.getcwd()
        assert os.path.isabs(base)
        path = os.path.join(base, path)
    return os.path.normpath(path)


def localpath(path, base=None, real=True):
    if base is None:
        base = os.getcwd()
    assert all(os.path.isabs(p) for p in (path, base))
    if real:
        # legal realpath
        path = os.path.realpath(path)
        base = os.path.realpath(base)
    if not path_startswith(path, base):
        return None
    return os.path.relpath(path, base)


def path_explode(p):
    """
    >>> path_explode('/')
    ['/']
    >>> path_explode('////')
    ['/']
    >>> path_explode('')
    []
    >>> path_explode('/1/2')
    ['/', '1', '2']
    >>> path_explode('/1///2//')
    ['/', '1', '2']
    """

    if p == '':
        return []

    p = os.path.normpath(p)

    components = []
    while True:
        (next_p, tail) = os.path.split(p)

        if p == next_p:
            if p != '':  # If the path is relative, empty string should not be added
                components.insert(0, p)  # this is likely to be '/'
            return components

        components.insert(0, tail)

        p = next_p


def path_startswith(path, prefix):
    """
    >>> path_startswith('/1/2/3', '/1')
    True
    >>> path_startswith('/1/2/3/', '/1/2')
    True
    >>> path_startswith('/1/2/3/', '1/2')
    False
    >>> path_startswith('/1/2/3/', '/5/1/2')
    False
    >>> path_startswith('/1/2/3/', '/1/2/3//////')
    True
    >>> path_startswith('/1/2/3/', '/1/2/3////4//')
    False
    """
    path_exp = path_explode(path)
    prefix_exp = path_explode(prefix)

    path_len = len(path_exp)
    prefix_len = len(prefix_exp)

    if prefix_len > path_len:
        return False

    return path_exp[:prefix_len] == prefix_exp


def path_prefixes(path):
    if not path:
        return
    path_s = []
    while path:
        path, tail = os.path.split(path)
        if tail:
            path_s.append(tail)
        else:
            break
    path_s.reverse()
    if path:
        yield path
    for n in six.moves.range(1, len(path_s) + 1):
        yield path + os.sep.join(path_s[:n])
