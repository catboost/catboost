import pathlib
import io
import os
import re
import sys
import warnings

from importlib.metadata import (
    Distribution,
    DistributionFinder,
    PackageNotFoundError,
    Prepared,
)
from importlib.resources.abc import Traversable

import __res

with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
    from importlib.abc import ResourceReader

ResourceReader.register(__res._ResfsResourceReader)

METADATA_NAME = re.compile("^Name: (.*)$", re.MULTILINE)


class ArcadiaResource(Traversable):
    def __init__(self, resfs_key):
        self.resfs_key = resfs_key

    def is_file(self):
        return True

    def is_dir(self):
        return False

    def open(self, mode="r", *args, **kwargs):
        data = __res.find(self.resfs_key.encode("utf-8"))
        if data is None:
            raise FileNotFoundError(self.resfs_key)

        stream = io.BytesIO(data)

        if "b" not in mode:
            stream = io.TextIOWrapper(stream, *args, **kwargs)

        return stream

    def joinpath(self, *name):
        raise RuntimeError("Cannot traverse into a resource")

    def iterdir(self):
        return iter(())

    @property
    def name(self):
        return os.path.basename(self.resfs_key)

    def __repr__(self):
        return f"ArcadiaResource({self.resfs_key!r})"


class ArcadiaResourceContainer(Traversable):
    def __init__(self, prefix):
        self.resfs_prefix = prefix

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def iterdir(self):
        seen = set()
        for key, path_without_prefix in __res.iter_keys(
            self.resfs_prefix.encode("utf-8")
        ):
            if b"/" in path_without_prefix:
                subdir = path_without_prefix.split(b"/", maxsplit=1)[0].decode("utf-8")
                if subdir not in seen:
                    seen.add(subdir)
                    yield ArcadiaResourceContainer(f"{self.resfs_prefix}{subdir}/")
            else:
                yield ArcadiaResource(key.decode("utf-8"))

    def open(self, *args, **kwargs):
        raise IsADirectoryError(self.resfs_prefix)

    @staticmethod
    def _flatten(compound_names):
        for name in compound_names:
            yield from name.split("/")

    def joinpath(self, *descendants):
        if not descendants:
            return self

        names = self._flatten(descendants)
        target = next(names)
        for traversable in self.iterdir():
            if traversable.name == target:
                if isinstance(traversable, ArcadiaResource):
                    return traversable
                else:
                    return traversable.joinpath(*names)

        raise FileNotFoundError("/".join(self._flatten(descendants)))

    @property
    def name(self):
        return os.path.basename(self.resfs_prefix[:-1])

    def __repr__(self):
        return f"ArcadiaResourceContainer({self.resfs_prefix!r})"


class ArcadiaDistribution(Distribution):
    def __init__(self, prefix):
        self._prefix = prefix
        self._path = pathlib.Path(prefix)

    def read_text(self, filename):
        data = __res.resfs_read(f"{self._prefix}{filename}")
        if data is not None:
            return data.decode("utf-8")

    read_text.__doc__ = Distribution.read_text.__doc__

    def locate_file(self, path):
        return self._path.parent / path


class ArcadiaMetadataFinder(DistributionFinder):
    prefixes = {}

    @classmethod
    def find_distributions(cls, context=DistributionFinder.Context()):
        found = cls._search_prefixes(context.name)
        return map(ArcadiaDistribution, found)

    @classmethod
    def _init_prefixes(cls):
        cls.prefixes.clear()

        for resource in __res.resfs_files():
            resource = resource.decode("utf-8")
            if not resource.endswith("METADATA"):
                continue
            data = __res.resfs_read(resource).decode("utf-8")
            metadata_name = METADATA_NAME.search(data)
            if metadata_name:
                metadata_name = Prepared(metadata_name.group(1))
                cls.prefixes[metadata_name.normalized] = resource[: -len("METADATA")]

    @classmethod
    def _search_prefixes(cls, name):
        if not cls.prefixes:
            cls._init_prefixes()

        if name:
            try:
                yield cls.prefixes[Prepared(name).normalized]
            except KeyError:
                raise PackageNotFoundError(name)
        else:
            yield from sorted(cls.prefixes.values())


# monkeypatch standart library
import importlib.metadata

importlib.metadata.MetadataPathFinder = ArcadiaMetadataFinder
