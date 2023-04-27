"""General mocking utilities"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import sys

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch # py2


class MultiPatch(object):
    def __init__(self, *patchers):
        self.patchers = patchers
    
    def __enter__(self):
        for p in self.patchers:
            p.start()
    
    def __exit__(self, *args):
        for p in self.patchers:
            p.stop()

darwin = MultiPatch(
    patch.object(os, 'name', 'posix'),
    patch.object(sys, 'platform', 'darwin'),
)

linux = MultiPatch(
    patch.object(os, 'name', 'posix'),
    patch.object(sys, 'platform', 'linux2'),
)

windows = MultiPatch(
    patch.object(os, 'name', 'nt'),
    patch.object(sys, 'platform', 'win32'),
)
