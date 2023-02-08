# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import abc
import binascii
import os
import sys
import warnings
from hashlib import sha384
from typing import Dict, Iterable

from hypothesis.configuration import mkdir_p, storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.utils.conventions import not_set

__all__ = [
    "DirectoryBasedExampleDatabase",
    "ExampleDatabase",
    "InMemoryExampleDatabase",
    "MultiplexedDatabase",
    "ReadOnlyDatabase",
]


def _usable_dir(path):
    """
    Returns True iff the desired path can be used as database path because
    either the directory exists and can be used, or its root directory can
    be used and we can make the directory as needed.
    """
    while not os.path.exists(path):
        # Loop terminates because the root dir ('/' on unix) always exists.
        path = os.path.dirname(path)
    return os.path.isdir(path) and os.access(path, os.R_OK | os.W_OK | os.X_OK)


def _db_for_path(path=None):
    if path is not_set:
        if os.getenv("HYPOTHESIS_DATABASE_FILE") is not None:  # pragma: no cover
            raise HypothesisException(
                "The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any "
                "effect.  Configure your database location via a settings profile instead.\n"
                "https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles"
            )

        path = storage_directory("examples")
        if not _usable_dir(path):  # pragma: no cover
            warnings.warn(
                HypothesisWarning(
                    "The database setting is not configured, and the default "
                    "location is unusable - falling back to an in-memory "
                    f"database for this session.  path={path!r}"
                )
            )
            return InMemoryExampleDatabase()
    if path in (None, ":memory:"):
        return InMemoryExampleDatabase()
    return DirectoryBasedExampleDatabase(str(path))


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        if self is ExampleDatabase:
            return _db_for_path(*args, **kwargs)
        return super().__call__(*args, **kwargs)


# This __call__ method is picked up by Sphinx as the signature of all ExampleDatabase
# subclasses, which is accurate, reasonable, and unhelpful.  Fortunately Sphinx
# maintains a list of metaclass-call-methods to ignore, and while they would prefer
# not to maintain it upstream (https://github.com/sphinx-doc/sphinx/pull/8262) we
# can insert ourselves here.
#
# This code only runs if Sphinx has already been imported; and it would live in our
# docs/conf.py except that we would also like it to work for anyone documenting
# downstream ExampleDatabase subclasses too.
if "sphinx" in sys.modules:
    try:
        from sphinx.ext.autodoc import _METACLASS_CALL_BLACKLIST

        _METACLASS_CALL_BLACKLIST.append("hypothesis.database._EDMeta.__call__")
    except Exception:
        pass


class ExampleDatabase(metaclass=_EDMeta):
    """An abstract base class for storing examples in Hypothesis' internal format.

    An ExampleDatabase maps each ``bytes`` key to many distinct ``bytes``
    values, like a ``Mapping[bytes, AbstractSet[bytes]]``.
    """

    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If this value is already present for this key, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.save")

    @abc.abstractmethod
    def fetch(self, key: bytes) -> Iterable[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f"{type(self).__name__}.fetch")

    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None:
        """Remove this value from this key.

        If this value is not present, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.delete")

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        """Move ``value`` from key ``src`` to key ``dest``. Equivalent to
        ``delete(src, value)`` followed by ``save(src, value)``, but may
        have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.save(src, value)
            return
        self.delete(src, value)
        self.save(dest, value)


class InMemoryExampleDatabase(ExampleDatabase):
    """A non-persistent example database, implemented in terms of a dict of sets.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self):
        self.data = {}

    def __repr__(self) -> str:
        return f"InMemoryExampleDatabase({self.data!r})"

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self.data.get(key, ())

    def save(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def delete(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))


def _hash(key):
    return sha384(key).hexdigest()[:16]


class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.  While the contents are fairly opaque, a
    ``DirectoryBasedExampleDatabase`` can be shared by checking the directory
    into version control, for example with the following ``.gitignore``::

        # Ignore files cached by Hypothesis...
        .hypothesis/*
        # except for the examples directory
        !.hypothesis/examples/

    Note however that this only makes sense if you also pin to an exact version of
    Hypothesis, and we would usually recommend implementing a shared database with
    a network datastore - see :class:`~hypothesis.database.ExampleDatabase`, and
    the :class:`~hypothesis.database.MultiplexedDatabase` helper.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.keypaths: Dict[str, str] = {}

    def __repr__(self) -> str:
        return f"DirectoryBasedExampleDatabase({self.path!r})"

    def _key_path(self, key):
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        directory = os.path.join(self.path, _hash(key))
        self.keypaths[key] = directory
        return directory

    def _value_path(self, key, value):
        return os.path.join(self._key_path(key), _hash(value))

    def fetch(self, key: bytes) -> Iterable[bytes]:
        kp = self._key_path(key)
        if not os.path.exists(kp):
            return
        for path in os.listdir(kp):
            try:
                with open(os.path.join(kp, path), "rb") as i:
                    yield i.read()
            except OSError:
                pass

    def save(self, key: bytes, value: bytes) -> None:
        # Note: we attempt to create the dir in question now. We
        # already checked for permissions, but there can still be other issues,
        # e.g. the disk is full
        mkdir_p(self._key_path(key))
        path = self._value_path(key, value)
        if not os.path.exists(path):
            suffix = binascii.hexlify(os.urandom(16)).decode("ascii")
            tmpname = path + "." + suffix
            with open(tmpname, "wb") as o:
                o.write(value)
            try:
                os.rename(tmpname, path)
            except OSError:  # pragma: no cover
                os.unlink(tmpname)
            assert not os.path.exists(tmpname)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return
        try:
            os.renames(self._value_path(src, value), self._value_path(dest, value))
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def delete(self, key: bytes, value: bytes) -> None:
        try:
            os.unlink(self._value_path(key, value))
        except OSError:
            pass


class ReadOnlyDatabase(ExampleDatabase):
    """A wrapper to make the given database read-only.

    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and
    ``move`` into silent no-ops.

    Note that this disables Hypothesis' automatic discarding of stale examples.
    It is designed to allow local machines to access a shared database (e.g. from CI
    servers), without propagating changes back from a local or in-development branch.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f"ReadOnlyDatabase({self._wrapped!r})"

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
        pass


class MultiplexedDatabase(ExampleDatabase):
    """A wrapper around multiple databases.

    Each ``save``, ``fetch``, ``move``, or ``delete`` operation will be run against
    all of the wrapped databases.  ``fetch`` does not yield duplicate values, even
    if the same value is present in two or more of the wrapped databases.

    This combines well with a :class:`ReadOnlyDatabase`, as follows:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase("/tmp/hypothesis/examples/")
        shared = CustomNetworkDatabase()

        settings.register_profile("ci", database=shared)
        settings.register_profile(
            "dev", database=MultiplexedDatabase(local, ReadOnlyDatabase(shared))
        )
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    So your CI system or fuzzing runs can populate a central shared database;
    while local runs on development machines can reproduce any failures from CI
    but will only cache their own failures locally and cannot remove examples
    from the shared database.
    """

    def __init__(self, *dbs: ExampleDatabase) -> None:
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return "MultiplexedDatabase({})".format(", ".join(map(repr, self._wrapped)))

    def fetch(self, key: bytes) -> Iterable[bytes]:
        seen = set()
        for db in self._wrapped:
            for value in db.fetch(key):
                if value not in seen:
                    yield value
                    seen.add(value)

    def save(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.save(key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.delete(key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.move(src, dest, value)
