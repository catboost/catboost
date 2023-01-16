import base64
import binascii
import yaml
import os

from six.moves.urllib import parse as urlparse
from six import iteritems

from ..base import PackageJson, BaseLockfile, LockfilePackageMeta, LockfilePackageMetaInvalidError


class PnpmLockfile(BaseLockfile):
    IMPORTER_KEYS = PackageJson.DEP_KEYS + ("specifiers",)

    def read(self):
        with open(self.path, "r") as f:
            self.data = yaml.load(f, Loader=yaml.CSafeLoader)

    def write(self, path=None):
        """
        :param path: path to store lockfile, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            yaml.dump(self.data, f, Dumper=yaml.CSafeDumper)

    def get_packages_meta(self):
        """
        Extracts packages meta from lockfile.
        :rtype: list of LockfilePackageMeta
        """
        packages = self.data.get("packages", {})

        return map(lambda x: _parse_package_meta(*x), iteritems(packages))

    def update_tarball_resolutions(self, fn):
        """
        :param fn: maps `LockfilePackageMeta` instance to new `resolution.tarball` value
        :type fn: lambda
        """
        packages = self.data.get("packages", {})

        for key, meta in iteritems(packages):
            meta["resolution"]["tarball"] = fn(_parse_package_meta(key, meta))
            packages[key] = meta

    def get_importers(self):
        """
        Returns "importers" section from the lockfile or creates similar structure from "dependencies" and "specifiers".
        :rtype: dict of dict of dict of str
        """
        importers = self.data.get("importers")
        if importers is not None:
            return importers

        importer = {k: self.data[k] for k in self.IMPORTER_KEYS if k in self.data}

        return ({".": importer} if importer else {})

    def merge(self, lf):
        """
        Merges two lockfiles:
        1. Converts the lockfile to monorepo-like lockfile with "importers" section instead of "dependencies" and "specifiers".
        2. Merges `lf`'s dependencies and specifiers to importers.
        3. Merges `lf`'s packages to the lockfile.
        :param lf: lockfile to merge
        :type lf: PnpmLockfile
        """
        importers = self.get_importers()
        build_path = os.path.dirname(self.path)

        for [importer, imports] in iteritems(lf.get_importers()):
            importer_path = os.path.normpath(os.path.join(os.path.dirname(lf.path), importer))
            importer_rel_path = os.path.relpath(importer_path, build_path)
            importers[importer_rel_path] = imports

        self.data["importers"] = importers

        for k in self.IMPORTER_KEYS:
            self.data.pop(k, None)

        packages = self.data.get("packages", {})
        for k, v in iteritems(lf.data.get("packages", {})):
            if k not in packages:
                packages[k] = v
        self.data["packages"] = packages


def _parse_package_meta(key, meta):
    """
    :param key: uniq package key from lockfile
    :type key: string
    :param meta: package meta dict from lockfile
    :type meta: dict
    :rtype: LockfilePackageMetaInvalidError
    """
    try:
        name, version = _parse_package_key(key)
        sky_id = _parse_sky_id_from_tarball_url(meta["resolution"]["tarball"])
        integrity_algorithm, integrity = _parse_package_integrity(meta["resolution"]["integrity"])
    except KeyError as e:
        raise TypeError("Invalid package meta for key {}, missing {} key".format(key, e))
    except LockfilePackageMetaInvalidError as e:
        raise TypeError("Invalid package meta for key {}, parse error: {}".format(key, e))

    return LockfilePackageMeta(name, version, sky_id, integrity, integrity_algorithm)


def _parse_package_key(key):
    """
    :param key: package key in format "/({scope}/)?{package_name}/{package_version}(_{peer_dependencies})?"
    :type key: string
    :return: tuple of scoped package name and version
    :rtype: (str, str)
    """
    try:
        tokens = key.split("/")[1:]
        version = tokens.pop().split("_", 1)[0]

        if len(tokens) < 1 or len(tokens) > 2:
            raise TypeError()
    except (IndexError, TypeError):
        raise LockfilePackageMetaInvalidError("Invalid package key")

    return ("/".join(tokens), version)


def _parse_sky_id_from_tarball_url(tarball_url):
    """
    :param tarball_url: tarball url
    :type tarball_url: string
    :return: sky id
    :rtype: string
    """
    if tarball_url.startswith("file:"):
        return ""

    rbtorrent_param = urlparse.parse_qs(urlparse.urlparse(tarball_url).query).get("rbtorrent")

    if rbtorrent_param is None:
        raise LockfilePackageMetaInvalidError("Missing rbtorrent param in tarball url {}".format(tarball_url))

    return "rbtorrent:{}".format(rbtorrent_param[0])


def _parse_package_integrity(integrity):
    """
    :param integrity: package integrity in format "{algo}-{base64_of_hash}"
    :type integrity: string
    :return: tuple of algorithm and hash (hex)
    :rtype: (str, str)
    """
    algo, hash_b64 = integrity.split("-", 1)

    try:
        hash_hex = binascii.hexlify(base64.b64decode(hash_b64))
    except TypeError as e:
        raise LockfilePackageMetaInvalidError("Invalid package integrity encoding, integrity: {}, error: {}".format(integrity, e))

    return (algo, hash_hex)
