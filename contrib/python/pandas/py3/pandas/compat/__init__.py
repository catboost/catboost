"""
compat
======

Cross-compatible functions for different versions of Python.

Other items:
* platform checker
"""
import os
import platform
import sys

from pandas._typing import F
from pandas.compat.numpy import (
    is_numpy_dev,
    np_version_under1p19,
    np_version_under1p20,
)
from pandas.compat.pyarrow import (
    pa_version_under1p01,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
)

PY39 = sys.version_info >= (3, 9)
PY310 = sys.version_info >= (3, 10)
PYPY = platform.python_implementation() == "PyPy"
IS64 = sys.maxsize > 2**32


def set_function_name(f: F, name: str, cls) -> F:
    """
    Bind the name/qualname attributes of the function.
    """
    f.__name__ = name
    f.__qualname__ = f"{cls.__name__}.{name}"
    f.__module__ = cls.__module__
    return f


def is_platform_little_endian() -> bool:
    """
    Checking if the running platform is little endian.

    Returns
    -------
    bool
        True if the running platform is little endian.
    """
    return sys.byteorder == "little"


def is_platform_windows() -> bool:
    """
    Checking if the running platform is windows.

    Returns
    -------
    bool
        True if the running platform is windows.
    """
    return sys.platform in ["win32", "cygwin"]


def is_platform_linux() -> bool:
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    return sys.platform == "linux"


def is_platform_mac() -> bool:
    """
    Checking if the running platform is mac.

    Returns
    -------
    bool
        True if the running platform is mac.
    """
    return sys.platform == "darwin"


def is_platform_arm() -> bool:
    """
    Checking if the running platform use ARM architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("arm64", "aarch64") or platform.machine().startswith(
        "armv"
    )


def is_ci_environment() -> bool:
    """
    Checking if running in a continuous integration environment by checking
    the PANDAS_CI environment variable.

    Returns
    -------
    bool
        True if the running in a continuous integration environment.
    """
    return os.environ.get("PANDAS_CI", "0") == "1"


def get_lzma_file():
    """
    Importing the `LZMAFile` class from the `lzma` module.

    Returns
    -------
    class
        The `LZMAFile` class from the `lzma` module.

    Raises
    ------
    RuntimeError
        If the `lzma` module was not imported correctly, or didn't exist.
    """
    try:
        import lzma
    except ImportError:
        raise RuntimeError(
            "lzma module not available. "
            "A Python re-install with the proper dependencies, "
            "might be required to solve this issue."
        )
    return lzma.LZMAFile


__all__ = [
    "is_numpy_dev",
    "np_version_under1p19",
    "np_version_under1p20",
    "pa_version_under1p01",
    "pa_version_under2p0",
    "pa_version_under3p0",
    "pa_version_under4p0",
]
