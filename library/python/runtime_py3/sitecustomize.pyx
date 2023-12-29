import pathlib
import io
import os
import re
import sys
import warnings

from importlib.metadata import Distribution, DistributionFinder, PackageNotFoundError, Prepared
from importlib.resources.abc import Traversable

import __res

with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
    from importlib.abc import ResourceReader

ResourceReader.register(__res._ResfsResourceReader)

METADATA_NAME = re.compile('^Name: (.*)$', re.MULTILINE)


class ArcadiaResourceHandle(Traversable):
    def __init__(self, key):
        self.resfs_key = key

    def is_file(self):
        return True

    def is_dir(self):
        return False

    def open(self, mode='r', *args, **kwargs):
        data = __res.find(self.resfs_key.encode("utf-8"))
        if data is None:
            raise FileNotFoundError(self.resfs_key)

        stream = io.BytesIO(data)

        if 'b' not in mode:
            stream = io.TextIOWrapper(stream, *args, **kwargs)

        return stream

    def joinpath(self, *name):
        raise RuntimeError("Cannot traverse into a resource")

    def iterdir(self):
        return iter(())

    @property
    def name(self):
        return os.path.basename(self.resfs_key)


class ArcadiaResourceContainer(Traversable):
    def __init__(self, prefix):
        self.resfs_prefix = prefix

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def iterdir(self):
        for key, path_without_prefix in __res.iter_keys(self.resfs_prefix.encode("utf-8")):
            if b"/" in path_without_prefix:
                name = path_without_prefix.decode("utf-8").split("/", maxsplit=1)[0]
                yield ArcadiaResourceContainer(f"{self.resfs_prefix}{name}/")
            else:
                yield ArcadiaResourceHandle(key.decode("utf-8"))

    def open(self, *args, **kwargs):
        raise IsADirectoryError(self.resfs_prefix)

    def joinpath(self, *descendants):
        if not descendants:
            return self

        return ArcadiaResourceHandle(os.path.join(self.resfs_prefix, *descendants))

    @property
    def name(self):
        return os.path.basename(self.resfs_prefix[:-1])


class ArcadiaDistribution(Distribution):

    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def _path(self):
        return pathlib.Path(self.prefix)

    def read_text(self, filename):
        data = __res.resfs_read(f'{self.prefix}{filename}')
        if data:
            return data.decode('utf-8')
    read_text.__doc__ = Distribution.read_text.__doc__

    def locate_file(self, path):
        return f'{self.prefix}{path}'


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
            resource = resource.decode('utf-8')
            if not resource.endswith('METADATA'):
                continue
            data = __res.resfs_read(resource).decode('utf-8')
            metadata_name = METADATA_NAME.search(data)
            if metadata_name:
                metadata_name = Prepared(metadata_name.group(1))
                cls.prefixes[metadata_name.normalized] = resource[:-len('METADATA')]

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
            for prefix in sorted(cls.prefixes.values()):
                yield prefix


# monkeypatch standart library
import importlib.metadata
importlib.metadata.MetadataPathFinder = ArcadiaMetadataFinder
