import os

import yatest.common
import __res


def is_relative_to(path, parent):
    """
    A polyfill for pathlib.Path.is_relative_to (string-based implementation).
    Does NOT involve filesystem operations.
    """
    path = os.path.normpath(path)
    parent = os.path.normpath(parent)

    if path == parent:
        return True

    if not parent.endswith(os.sep):
        parent += os.sep

    return path.startswith(parent)


def relative_to(path, parent):
    """
    A polyfill for pathlib.Path.relative_to (string-based implementation).
    Does NOT involve filesystem operations.
    """
    path = os.path.normpath(path)
    parent = os.path.normpath(parent)

    if path == parent:
        return ''

    if not parent.endswith(os.sep):
        parent += os.sep

    # Check if path starts with parent
    if not path.startswith(parent):
        raise ValueError('Path {} is not relative to {}', path, parent)

    return path[len(parent) :]


def get_module_file_path(module_name):
    """
    Get the file path for a module without importing it.
    For synthesized empty __init__.py modules, returns module_name as a fake file path.
    """
    return __res.importer.get_filename(module_name)


_BASE_PATHS = None


def get_proper_module_path(module_name):
    """
    Get the file path for a test module relative to the repository root.
    For synthesized empty __init__.py modules, returns module_name as a fake file path.
    """
    global _BASE_PATHS
    if _BASE_PATHS is None:
        _BASE_PATHS = (
            str(yatest.common.source_path()),
            str(yatest.common.work_path()),
            str(yatest.common.build_path()),
        )

    path = get_module_file_path(module_name)

    if not os.path.isabs(path):
        if path == module_name:
            # Synthesized __init__.py module.
            return None
        return path

    for base_path in _BASE_PATHS:
        if is_relative_to(path, base_path):
            return relative_to(path, base_path)

    raise ValueError(
        'Failed to resolve module "{}" with path="{}" against known base paths'.format(module_name, path),
    )
