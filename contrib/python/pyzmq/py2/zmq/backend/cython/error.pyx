"""0MQ Error classes and functions."""

#
#    Copyright (c) 2010-2011 Brian E. Granger & Min Ragan-Kelley
#
#    This file is part of pyzmq.
#
#    pyzmq is free software; you can redistribute it and/or modify it under
#    the terms of the Lesser GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    pyzmq is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    Lesser GNU General Public License for more details.
#
#    You should have received a copy of the Lesser GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from cpython.version cimport PY_MAJOR_VERSION
from .libzmq cimport zmq_strerror, zmq_errno as zmq_errno_c

def strerror(int errno):
    """strerror(errno)

    Return the error string given the error number.
    """
    # This is a const char *
    str_e = zmq_strerror(errno)
    if PY_MAJOR_VERSION <= 2:
        # Python 2: str is bytes, so we already have the right type
        return str_e
    else:
        # Python 3: decode bytes to unicode str
        return str_e.decode()

def zmq_errno():
    """zmq_errno()
    
    Return the integer errno of the most recent zmq error.
    """
    return zmq_errno_c()

__all__ = ['strerror', 'zmq_errno']
