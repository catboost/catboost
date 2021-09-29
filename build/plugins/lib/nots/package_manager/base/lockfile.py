import os

from abc import ABCMeta, abstractmethod
from six import add_metaclass


class LockfilePackageMeta(object):
    """
    Basic struct representing package meta from lockfile.
    """
    __slots__ = ("name", "version", "sky_id", "integrity", "integrity_algorithm", "tarball_path")

    @staticmethod
    def from_str(s):
        return LockfilePackageMeta(*s.strip().split(" "))

    def __init__(self, name, version, sky_id, integrity, integrity_algorithm):
        self.name = name
        self.version = version
        self.sky_id = sky_id
        self.integrity = integrity
        self.integrity_algorithm = integrity_algorithm
        self.tarball_path = "{}-{}.tgz".format(name, version)

    def to_str(self):
        return " ".join([self.name, self.version, self.sky_id, self.integrity, self.integrity_algorithm])


class LockfilePackageMetaInvalidError(RuntimeError):
    pass


@add_metaclass(ABCMeta)
class BaseLockfile(object):
    @classmethod
    def load(cls, path):
        """
        :param path: lockfile path
        :type path: str
        :rtype: BaseLockfile
        """
        pj = cls(path)
        pj.read()

        return pj

    def __init__(self, path):
        if not os.path.isabs(path):
            raise TypeError("Absolute path required, given: {}".format(path))

        self.path = path
        self.data = None

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, path=None):
        pass

    @abstractmethod
    def get_packages_meta(self):
        pass

    @abstractmethod
    def update_tarball_resolutions(self, fn):
        pass
