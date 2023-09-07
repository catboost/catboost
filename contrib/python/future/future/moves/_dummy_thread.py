from __future__ import absolute_import
from future.utils import PY3

if PY3:
    try:
        from _dummy_thread import *
    except ImportError:
        from _thread import *
else:
    __future_module__ = True
    from dummy_thread import *
