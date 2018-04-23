from __res import find, count, key_by_index
from __res import resfs_resolve, resfs_read, resfs_src, resfs_files  # noqa


def iterkeys(prefix='', strip_prefix=False):
    for i in xrange(count()):
        key = key_by_index(i)
        if key.startswith(prefix):
            if strip_prefix:
                key = key[len(prefix):]
            yield key


def iteritems(prefix='', strip_prefix=False):
    for key in iterkeys(prefix=prefix):
        value = find(key)
        if strip_prefix:
            key = key[len(prefix):]
        yield key, value
