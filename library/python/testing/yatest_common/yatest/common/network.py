# coding=utf-8
"""
DEPRECATED: This module is deprecated and kept only for backward compatibility.

Please use library.python.port_manager instead:
    from library.python.port_manager import PortManager

This module will be removed in future versions.
"""

import warnings

warnings.warn(
    "yatest.common.network is deprecated, use library.python.port_manager instead",
    DeprecationWarning,
    stacklevel=2,
)

from library.python.port_manager import *  # noqa
