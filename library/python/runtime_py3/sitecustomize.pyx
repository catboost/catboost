import sys

import __res

from importlib.abc import ResourceReader
from importlib.metadata import Distribution, DistributionFinder

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

    def find_distributions(self, context=DistributionFinder.Context()):
        found = self._search_prefixes(context.name)
        return map(ArcadiaDistribution, found)

    @classmethod
    def _init_prefixes(cls):
        cls.prefixes = {}

        for resource in __res.resfs_files():
            resource = resource.decode('utf-8')
            if not resource.endswith('top_level.txt'):
                continue
            data = __res.resfs_read(resource).decode('utf-8')
            for top_level in data.split('\n'):
                if top_level:
                    cls.prefixes[top_level] = resource[:-len('top_level.txt')]

    @classmethod
    def _search_prefixes(cls, name):
        if not cls.prefixes:
            cls._init_prefixes()

        if name:
            yield cls.prefixes[name.replace('-', '_')]
        else:
            for prefix in sorted(set(cls.prefixes.values())):
                yield prefix


sys.meta_path.append(ArcadiaMetadataFinder())
