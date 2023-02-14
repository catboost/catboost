import six
from six.moves import collections_abc


def flatten(coll):
    for i in coll:
        if isinstance(i, collections_abc.Iterable) and not isinstance(i, six.string_types):
            for subc in flatten(i):
                yield subc
        else:
            yield i
