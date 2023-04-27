"""PyZMQ and 0MQ version functions."""

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

from .libzmq cimport _zmq_version

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def zmq_version_info():
    """zmq_version_info()

    Return the version of ZeroMQ itself as a 3-tuple of ints.
    """
    cdef int major, minor, patch
    _zmq_version(&major, &minor, &patch)
    return (major, minor, patch)


__all__ = ['zmq_version_info']

