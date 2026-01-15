import functools
import io
import os
import pathlib
import re
import stat
import sys
import typing

from importlib.metadata import (
    Distribution,
    DistributionFinder,
    Prepared,
)
from importlib.resources.abc import Traversable

from __res import find, iter_keys, resfs_files, resfs_read

METADATA_NAME = re.compile("^Name: (.*)$", re.MULTILINE)

try:
    BINARY_STAT: typing.Final[os.stat_result] = os.stat(sys.executable)
except FileNotFoundError:
    BINARY_STAT: typing.Final[os.stat_result] = os.stat_result(
        (
            0,  # st_mode
            1,  # st_ino
            0,  # st_dev
            1,  # st_nlink
            0,  # st_uid
            0,  # st_gid
            0,  # st_size
            0,  # st_atime
            0,  # st_mtime
            0,  # st_ctime
        ),
        {
            "st_atime_ns": 0,
            "st_mtime_ns": 0,
            "st_ctime_ns": 0,
        },
    )

RESOURCE_DIRECTORY_STAT: typing.Final[os.stat_result] = os.stat_result(
    (
        stat.S_IFDIR | 0o755,  # st_mode
        1,  # st_ino
        0,  # st_dev
        1,  # st_nlink
        BINARY_STAT.st_uid,  # st_uid
        BINARY_STAT.st_gid,  # st_gid
        0,  # st_size
        BINARY_STAT.st_atime,  # st_atime
        BINARY_STAT.st_mtime,  # st_mtime
        BINARY_STAT.st_ctime,  # st_ctime
    ),
    {
        "st_atime_ns": BINARY_STAT.st_atime_ns,
        "st_mtime_ns": BINARY_STAT.st_mtime_ns,
        "st_ctime_ns": BINARY_STAT.st_ctime_ns,
    },
)


class ArcadiaTraversable(Traversable):
    def __init__(self, resfs) -> None:
        self._resfs = str(resfs)
        self._path = pathlib.Path(resfs)

    def __eq__(self, other) -> bool:
        if isinstance(other, ArcadiaTraversable):
            return self._path == other._path
        raise NotImplementedError

    def __lt__(self, other) -> bool:
        if isinstance(other, ArcadiaTraversable):
            return self._path < other._path
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self._path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._resfs!r})"

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def stem(self) -> str:
        return self._path.stem

    def with_suffix(self, suffix: str) -> typing.Self:
        return type(self)(self._path.with_suffix(suffix))

    def lstat(self) -> os.stat_result:
        return self.stat()

    def relative_to(self, other):
        if isinstance(other, ArcadiaTraversable):
            return self._path.relative_to(other._path)
        raise NotImplementedError

    def resolve(self, strict: bool = False) -> typing.Self:
        return self


class ArcadiaResource(ArcadiaTraversable):
    def is_file(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return False

    def open(self, mode="r", *args, **kwargs):
        data = self.data
        if data is None:
            raise FileNotFoundError(self._resfs)

        stream = io.BytesIO(data)

        if "b" not in mode:
            stream = io.TextIOWrapper(stream, *args, **kwargs)

        return stream

    def joinpath(self, *name) -> ArcadiaTraversable:
        raise RuntimeError("Cannot traverse into a resource")

    def iterdir(self):
        return iter(())

    @functools.cached_property
    def data(self) -> bytes | None:
        return find(self._resfs.encode("utf-8"))

    def stat(self) -> os.stat_result:
        data = self.data
        if data is None:
            raise FileNotFoundError(self._resfs)

        return os.stat_result(
            (
                stat.S_IFREG | 0o644,  # st_mode
                1,  # st_ino
                0,  # st_dev
                1,  # st_nlink
                BINARY_STAT.st_uid,  # st_uid
                BINARY_STAT.st_gid,  # st_gid
                len(data),  # st_size
                BINARY_STAT.st_atime,  # st_atime
                BINARY_STAT.st_mtime,  # st_mtime
                BINARY_STAT.st_ctime,  # st_ctime
            ),
            {
                "st_atime_ns": BINARY_STAT.st_atime_ns,
                "st_mtime_ns": BINARY_STAT.st_mtime_ns,
                "st_ctime_ns": BINARY_STAT.st_ctime_ns,
            },
        )


class ArcadiaResourceContainer(ArcadiaTraversable):
    def is_dir(self) -> bool:
        return True

    def is_file(self) -> bool:
        return False

    def iterdir(self):
        seen = set()
        for key, path_without_prefix in iter_keys(self._resfs.encode("utf-8")):
            if b"/" in path_without_prefix:
                subdir = path_without_prefix.split(b"/", maxsplit=1)[0].decode("utf-8")
                if subdir not in seen:
                    seen.add(subdir)
                    yield ArcadiaResourceContainer(f"{self._resfs}{subdir}/")
            else:
                yield ArcadiaResource(key.decode("utf-8"))

    def open(self, *args, **kwargs):
        raise IsADirectoryError(self._resfs)

    @staticmethod
    def _flatten(compound_names):
        for name in compound_names:
            yield from str(name).split("/")

    def joinpath(self, *descendants) -> ArcadiaTraversable:
        if not descendants:
            return self

        names = self._flatten(descendants)
        target = next(names)
        if target == ".":
            return self
        for traversable in self.iterdir():
            if traversable.name == target:
                if isinstance(traversable, ArcadiaResource):
                    return traversable
                else:
                    return traversable.joinpath(*names)

        raise FileNotFoundError("/".join(self._flatten(descendants)))

    @staticmethod
    def stat() -> os.stat_result:
        return RESOURCE_DIRECTORY_STAT


class ArcadiaDistribution(Distribution):
    def __init__(self, prefix):
        self._prefix = prefix
        self._path = pathlib.Path(prefix)

    def read_text(self, filename):
        data = resfs_read(f"{self._prefix}{filename}")
        if data is not None:
            return data.decode("utf-8")

    read_text.__doc__ = Distribution.read_text.__doc__

    def locate_file(self, path):
        return self._path.parent / path


class MetadataArcadiaFinder(DistributionFinder):
    prefixes = {}

    @classmethod
    def find_distributions(cls, context=DistributionFinder.Context()):
        found = cls._search_prefixes(context.name)
        return map(ArcadiaDistribution, found)

    @classmethod
    def _init_prefixes(cls):
        cls.prefixes.clear()

        for resource in resfs_files():
            resource = resource.decode("utf-8")
            if not resource.endswith("METADATA"):
                continue
            data = resfs_read(resource).decode("utf-8")
            metadata_name = METADATA_NAME.search(data)
            if metadata_name:
                cls.prefixes[Prepared(metadata_name.group(1)).normalized] = resource.removesuffix("METADATA")

    @classmethod
    def _search_prefixes(cls, name):
        if not cls.prefixes:
            cls._init_prefixes()

        if name:
            try:
                yield cls.prefixes[Prepared(name).normalized]
            except KeyError:
                pass
        else:
            yield from sorted(cls.prefixes.values())
