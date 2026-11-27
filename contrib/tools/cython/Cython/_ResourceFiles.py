"""Builtin compiler data paths; project files continue to use the filesystem."""

import io
import os
import posixpath
import sys

PREFIX = "/__cython_resources__/"


def is_resource_path(path):
    """Recognize the compiler namespace on both POSIX and Windows paths."""
    return path.replace("\\", "/").startswith(PREFIX)


def resource_name(path):
    name = posixpath.normpath(path.replace("\\", "/"))
    if not name.startswith(PREFIX):
        raise FileNotFoundError(path)
    return "cython/" + name[len(PREFIX) :]


def resource_exists(path):
    import __res as resource

    return resource.resfs_src(resource_name(path), resfs_file=True) is not None


def open_resource(path):
    import __res as resource

    data = resource.resfs_read(resource_name(path), builtin=True)
    if data is None:
        raise FileNotFoundError(path)
    return io.BytesIO(data)


def list_directory(path):
    if not is_resource_path(path):
        return sorted(os.listdir(path))
    import __res as resource

    prefix = resource_name(path) + "/"
    return sorted({name.decode("utf-8")[len(prefix):].split("/", 1)[0] for name in resource.resfs_files(prefix)})


def data_root(directory):
    if getattr(sys, "is_standalone_binary", False):
        return PREFIX + "Cython/" + directory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), directory))


def include_root(path, source_root=None):
    if getattr(sys, "is_standalone_binary", False):
        if not source_root and os.path.isdir(path):
            return path
        normalized = "/" + path.replace("\\", "/").strip("/")
        root_prefix = None
        if source_root:
            normalized = "/" + os.path.abspath(path).replace("\\", "/").strip("/")
            root_prefix = ("/" + os.path.abspath(source_root).replace("\\", "/").strip("/")).rstrip("/")
        for supplier in ("cython", "cython_py2"):
            marker = "/contrib/tools/" + supplier + "/Cython/Includes"
            if root_prefix is None:
                index = normalized.find(marker)
            else:
                # Project paths can contain the same component names below their own directory.
                index = len(root_prefix) if normalized.startswith(root_prefix + marker) else -1
            if index >= 0 and (index + len(marker) == len(normalized) or normalized[index + len(marker)] == "/"):
                return data_root("Includes") + normalized[index + len(marker) :]
    return path
