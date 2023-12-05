import warnings

from ._distutils import _modified


def __getattr__(name):
    if name not in ['newer_pairwise_group']:
        raise AttributeError(name)
    warnings.warn(
        "dep_util is Deprecated. Use functions from setuptools.modified instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_modified, name)
