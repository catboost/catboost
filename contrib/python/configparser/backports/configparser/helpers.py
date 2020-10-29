#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

try:
    from collections import UserDict
except ImportError:
    from UserDict import UserDict

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

try:
    import pathlib
except ImportError:
    pathlib = None

from io import open
import sys

try:
    from thread import get_ident
except ImportError:
    try:
        from _thread import get_ident
    except ImportError:
        from _dummy_thread import get_ident


__all__ = ['UserDict', 'OrderedDict', 'open']


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

native_str = str
str = type('str')


def from_none(exc):
    """raise from_none(ValueError('a')) == raise ValueError('a') from None"""
    exc.__cause__ = None
    exc.__suppress_context__ = True
    return exc


# from reprlib 3.2.1
def recursive_repr(fillvalue='...'):
    'Decorator to make a repr function return fillvalue for a recursive call'

    def decorating_function(user_function):
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
        return wrapper

    return decorating_function


# from collections 3.2.1
class _ChainMap(MutableMapping):
    ''' A ChainMap groups multiple dicts (or other mappings) together
    to create a single, updateable view.

    The underlying mappings are stored in a list.  That list is public and can
    accessed or updated using the *maps* attribute.  There is no other state.

    Lookups search the underlying mappings successively until a key is found.
    In contrast, writes, updates, and deletions only operate on the first
    mapping.

    '''

    def __init__(self, *maps):
        '''Initialize a ChainMap by setting *maps* to the given mappings.
        If no mappings are provided, a single empty dictionary is used.

        '''
        self.maps = list(maps) or [{}]  # always at least one map

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                # can't use 'key in mapping' with defaultdict
                return mapping[key]
            except KeyError:
                pass
        # support subclasses that define __missing__
        return self.__missing__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __len__(self):
        # reuses stored hash values if possible
        return len(set().union(*self.maps))

    def __iter__(self):
        return iter(set().union(*self.maps))

    def __contains__(self, key):
        return any(key in m for m in self.maps)

    @recursive_repr()
    def __repr__(self):
        return '{0.__class__.__name__}({1})'.format(
            self, ', '.join(map(repr, self.maps))
        )

    @classmethod
    def fromkeys(cls, iterable, *args):
        'Create a ChainMap with a single dict created from the iterable.'
        return cls(dict.fromkeys(iterable, *args))

    def copy(self):
        """
        New ChainMap or subclass with a new copy of
        maps[0] and refs to maps[1:]
        """
        return self.__class__(self.maps[0].copy(), *self.maps[1:])

    __copy__ = copy

    def new_child(self):  # like Django's Context.push()
        'New ChainMap with a new dict followed by all previous maps.'
        return self.__class__({}, *self.maps)

    @property
    def parents(self):  # like Django's Context.pop()
        'New ChainMap from maps[1:].'
        return self.__class__(*self.maps[1:])

    def __setitem__(self, key, value):
        self.maps[0][key] = value

    def __delitem__(self, key):
        try:
            del self.maps[0][key]
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def popitem(self):
        """
        Remove and return an item pair from maps[0].
        Raise KeyError is maps[0] is empty.
        """
        try:
            return self.maps[0].popitem()
        except KeyError:
            raise KeyError('No keys found in the first mapping.')

    def pop(self, key, *args):
        """
        Remove *key* from maps[0] and return its value.
        Raise KeyError if *key* not in maps[0].
        """

        try:
            return self.maps[0].pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def clear(self):
        'Clear maps[0], leaving maps[1:] intact.'
        self.maps[0].clear()


try:
    from collections import ChainMap
except ImportError:
    ChainMap = _ChainMap


_ABC = getattr(
    abc,
    'ABC',
    # Python 3.3 compatibility
    abc.ABCMeta(native_str('__ABC'), (object,), dict(__metaclass__=abc.ABCMeta)),
)


class _PathLike(_ABC):

    """Abstract base class for implementing the file system path protocol."""

    @abc.abstractmethod
    def __fspath__(self):
        """Return the file system path representation of the object."""
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return bool(
            hasattr(subclass, '__fspath__')
            # workaround for Python 3.5
            or pathlib
            and issubclass(subclass, pathlib.Path)
        )


PathLike = getattr(os, 'PathLike', _PathLike)


def _fspath(path):
    """Return the path representation of a path-like object.

    If str or bytes is passed in, it is returned unchanged. Otherwise the
    os.PathLike interface is used to get the path representation. If the
    path representation is not str or bytes, TypeError is raised. If the
    provided path is not str, bytes, or os.PathLike, TypeError is raised.
    """
    if isinstance(path, (str, bytes)):
        return path

    if not hasattr(path, '__fspath__') and isinstance(path, pathlib.Path):
        # workaround for Python 3.5
        return str(path)

    # Work from the object's type to match method resolution of other magic
    # methods.
    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)
    except AttributeError:

        if hasattr(path_type, '__fspath__'):
            raise
        else:
            raise TypeError(
                "expected str, bytes or os.PathLike object, "
                "not " + path_type.__name__
            )
    if isinstance(path_repr, (str, bytes)):
        return path_repr
    else:
        raise TypeError(
            "expected {}.__fspath__() to return str or bytes, "
            "not {}".format(path_type.__name__, type(path_repr).__name__)
        )


fspath = getattr(os, 'fspath', _fspath)
