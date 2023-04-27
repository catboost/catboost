from __future__ import absolute_import

import re, sys

IS_PYPY = 'PyPy' in sys.version

import mock.mock as _mock
from mock.mock import *

__version__ = '4.0.3'
version_info = tuple(int(p) for p in
                     re.match(r'(\d+).(\d+).(\d+)', __version__).groups())


__all__ = ('__version__', 'version_info') + _mock.__all__
