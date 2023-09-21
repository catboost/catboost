from __res import find as __find, count, key_by_index, resfs_files as __resfs_files
from __res import resfs_read, resfs_resolve, resfs_src  # noqa

import six


def iterkeys(prefix="", strip_prefix=False):
    decode = lambda s: s  # noqa: E731
    if isinstance(prefix, six.text_type):
        prefix = prefix.encode("utf-8")
        decode = lambda s: s.decode("utf-8")  # noqa: E731

    for i in six.moves.range(count()):
        key = key_by_index(i)
        if key.startswith(prefix):
            if strip_prefix:
                key = key[len(prefix) :]
            yield decode(key)


def itervalues(prefix=b""):
    for key in iterkeys(prefix=prefix):
        value = find(key)
        yield value


def iteritems(prefix="", strip_prefix=False):
    for key in iterkeys(prefix=prefix):
        value = find(key)
        if strip_prefix:
            key = key[len(prefix) :]
        yield key, value


def resfs_file_exists(path):
    return resfs_src(path, resfs_file=True) is not None


def resfs_files(prefix=""):
    decode = lambda s: s  # noqa: E731
    if isinstance(prefix, six.text_type):
        decode = lambda s: s.decode("utf-8")  # noqa: E731
    return [decode(s) for s in __resfs_files(prefix=prefix)]


def find(path):
    if isinstance(path, six.text_type):
        path = path.encode("utf-8")
    return __find(path)
