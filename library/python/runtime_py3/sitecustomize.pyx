import sys

import __res

from importlib.abc import ResourceReader
from importlib.metadata import Distribution, DistributionFinder, PackageNotFoundError

ResourceReader.register(__res._ResfsResourceReader)


class ArcadiaDistribution(Distribution):

    def __init__(self, prefix):
        self.prefix = prefix

    def read_text(self, filename):
        data = __res.resfs_read(f'{self.prefix}{filename}')
        if data:
            return data.decode('utf-8')

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
            if not resource.endswith('top_level.txt'):
                continue
            data = __res.resfs_read(resource).decode('utf-8')
            for top_level in data.split('\n'):
                if top_level:
                    cls.prefixes[top_level.lower()] = resource[:-len('top_level.txt')]

    @classmethod
    def _search_prefixes(cls, name):
        if not cls.prefixes:
            cls._init_prefixes()

        if name:
            try:
                yield cls.prefixes[name.lower().replace('-', '_')]
            except KeyError:
                raise PackageNotFoundError(name)
        else:
            for prefix in sorted(set(cls.prefixes.values())):
                yield prefix


# monkeypatch standart library
import importlib.metadata
importlib.metadata.MetadataPathFinder = ArcadiaMetadataFinder
