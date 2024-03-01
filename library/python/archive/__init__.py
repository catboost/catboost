import errno
import logging
import os
import random
import shutil
import stat
import string
import sys

import six

import libarchive
import libarchive._libarchive as _libarchive

from pathlib2 import PurePath

logger = logging.getLogger(__name__)

GZIP = "gzip"
ZSTD = "zstd"

ENCODING = "utf-8"


class ConfigureError(Exception):
    pass


class Level(object):
    def __init__(self, level):
        self.level = level


class Compression(object):
    Fast = Level(1)
    Default = Level(2)
    Best = Level(3)


def get_compression_level(filter_name, level):
    if level is None or not filter_name:
        return None
    elif isinstance(level, Level):
        level = {
            GZIP: {
                Compression.Fast: 1,
                Compression.Default: 6,
                Compression.Best: 9,
            },
            ZSTD: {
                Compression.Fast: 1,
                Compression.Default: 3,
                Compression.Best: 22,
            },
        }[filter_name][level]
    return level


def encode(value, encoding):
    return value.encode(encoding)


def extract_tar(tar_file_path, output_dir, strip_components=None, fail_on_duplicates=True):
    output_dir = encode(output_dir, ENCODING)
    _make_dirs(output_dir)
    with libarchive.Archive(tar_file_path, mode="rb") as tarfile:
        for e in tarfile:
            p = _strip_prefix(e.pathname, strip_components)
            if not p:
                continue
            dest = os.path.join(output_dir, encode(p, ENCODING))
            if e.pathname.endswith("/") or e.isdir():
                _make_dirs(dest)
                continue

            if strip_components and fail_on_duplicates:
                if os.path.exists(dest):
                    raise Exception(
                        "The file {} is duplicated because of strip_components={}".format(dest, strip_components)
                    )

            _make_dirs(os.path.dirname(dest))

            if e.ishardlink():
                src = os.path.join(output_dir, _strip_prefix(e.hardlink, strip_components))
                _hardlink(src, dest)
                continue
            if e.issym():
                src = _strip_prefix(e.linkname, strip_components)
                _symlink(src, dest)
                continue

            with open(dest, "wb") as f:
                if hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), e.mode & 0o7777)
                libarchive.call_and_check(
                    _libarchive.archive_read_data_into_fd,
                    tarfile._a,
                    tarfile._a,
                    f.fileno(),
                )


def _strip_prefix(path, strip_components):
    if not strip_components:
        return path
    p = PurePath(path)
    stripped = str(p.relative_to(*p.parts[:strip_components]))
    return '' if stripped == '.' else stripped


def tar(
    paths,
    output,
    compression_filter=None,
    compression_level=None,
    fixed_mtime=None,
    onerror=None,
    postprocess=None,
    dereference=False,
):
    if isinstance(paths, six.string_types):
        paths = [paths]

    if isinstance(output, six.string_types):
        temp_tar_path, stream = (
            output + "." + "".join(random.sample(string.ascii_lowercase, 8)),
            None,
        )
    else:
        temp_tar_path, stream = None, output

    compression_level = get_compression_level(compression_filter, compression_level)

    try:
        if compression_filter:
            filter_name = compression_filter
            if compression_level is not None:
                filter_opts = {"compression-level": str(compression_level)}
            else:
                filter_opts = {}
            # force gzip don't store mtime of the original file being compressed (http://www.gzip.org/zlib/rfc-gzip.html#file-format)
            if fixed_mtime is not None and compression_filter == GZIP:
                filter_opts["timestamp"] = ""
        else:
            filter_name = filter_opts = None

        with libarchive.Archive(
            stream or temp_tar_path,
            mode="wb",
            format="gnu",
            filter=filter_name,
            filter_opts=filter_opts,
            fixed_mtime=fixed_mtime,
        ) as tarfile:
            # determine order if fixed_mtime is specified to produce stable archive
            paths = paths if fixed_mtime is None else sorted(paths)

            for p in paths:
                if type(p) == tuple:
                    path, arcname = p
                else:
                    path, arcname = p, os.path.basename(p)

                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path, followlinks=dereference):
                        if fixed_mtime is None:
                            entries = dirs + files
                        else:
                            entries = sorted(dirs) + sorted(files)

                        reldir = os.path.relpath(root, path)
                        for f in entries:
                            _writepath(
                                tarfile,
                                os.path.join(root, f),
                                os.path.normpath(os.path.join(arcname, reldir, f)),
                                onerror,
                                postprocess,
                                dereference,
                            )
                else:
                    if not os.path.exists(path):
                        raise OSError("Specified path doesn't exist: {}".format(path))
                    _writepath(tarfile, path, arcname, onerror, postprocess, dereference)

        if temp_tar_path:
            os.rename(temp_tar_path, output)
    except Exception:
        if temp_tar_path and os.path.exists(temp_tar_path):
            os.remove(temp_tar_path)
        raise


def _writepath(tarfile, src, dst, onerror, postprocess, dereference):
    def tar_writepath(src, dst):
        st = os.lstat(src)
        if stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
            if dereference and stat.S_ISLNK(st.st_mode):
                src = os.path.realpath(src)

            tarfile.writepath(src, dst)

            if postprocess:
                postprocess(src, dst, st.st_mode)
        else:
            logger.debug("Skipping non-regular file '%s' (stat: %s)", src, st)

    try:
        return tar_writepath(src, dst)
    except Exception as e:
        if isinstance(e, OSError) and e.errno == errno.ENOENT:
            logger.debug(
                "Skipping missing file '%s' - looks like directory content has changed during archiving",
                src,
            )
            return

        if onerror:
            if onerror(src, dst, sys.exc_info()):
                return tar_writepath(src, dst)
        else:
            raise


def check_tar(tar_file_path):
    if os.path.isfile(tar_file_path) or os.path.islink(tar_file_path):
        return libarchive.is_archive(tar_file_path)
    return False


def _make_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(path):
            raise


def _hardlink(src, dst):
    if hasattr(os, "link"):
        os.link(src, dst)
    else:
        shutil.copyfile(src, dst)


def _symlink(src, dst):
    if hasattr(os, "symlink"):
        os.symlink(src, dst)
    else:
        # Windows specific case - we cannot copy file right now,
        # because it doesn't exist yet (and would be met later in the archive) or symlink is broken.
        # Act like tar and tarfile - skip such symlinks
        if os.path.exists(src):
            shutil.copytree(src, dst)


def get_archive_filter_name(filename):
    filters = libarchive.get_archive_filter_names(filename)
    # https://a.yandex-team.ru/arc/trunk/arcadia/contrib/libs/libarchive/libarchive/archive_read.c?rev=5800047#L522
    assert filters[-1] == "none", filters
    if len(filters) == 1:
        return None
    if len(filters) == 2:
        return filters[0]
    raise Exception("Archive has chain of filter: {}".format(filters))
