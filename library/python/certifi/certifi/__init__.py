import ssl

if hasattr(ssl, "builtin_cadata"):
    from .binary import where
else:
    from .source import where

__all__ = ["where", "__version__"]

__version__ = "2020.04.05.2"
