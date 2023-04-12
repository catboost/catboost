import pathlib
import re
import sys

import __res

from importlib.abc import ResourceReader
from importlib.metadata import Distribution, DistributionFinder, PackageNotFoundError, Prepared

ResourceReader.register(__res._ResfsResourceReader)

METADATA_NAME = re.compile('^Name: (.*)$', re.MULTILINE)


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
