"""Import basic exposure of libzmq C API as a backend"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import os
import platform
import sys

from .select import public_api, select_backend

if 'PYZMQ_BACKEND' in os.environ:
    backend = os.environ['PYZMQ_BACKEND']
    if backend in ('cython', 'cffi'):
        backend = 'zmq.backend.%s' % backend
    _ns = select_backend(backend)
else:
    # default to cython, fallback to cffi
    # (reverse on PyPy)
    if platform.python_implementation() == 'PyPy':
        first, second = ('zmq.backend.cffi', 'zmq.backend.cython')
    else:
        first, second = ('zmq.backend.cython', 'zmq.backend.cffi')

    try:
        _ns = select_backend(first)
    except Exception:
        exc_info = sys.exc_info()
        exc = exc_info[1]
        try:
            _ns = select_backend(second)
        except ImportError:
            # prevent 'During handling of the above exception...' on py3
            # can't use `raise ... from` on Python 2
            if hasattr(exc, '__cause__'):
                exc.__cause__ = None
            # raise the *first* error, not the fallback
            from zmq.utils.sixcerpt import reraise
            reraise(*exc_info)

globals().update(_ns)

__all__ = public_api
