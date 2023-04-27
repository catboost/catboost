"""0MQ utils."""

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

from .libzmq cimport zmq_curve_keypair, zmq_curve_public, zmq_has

from zmq.error import ZMQError, _check_rc, _check_version


def has(capability):
    """Check for zmq capability by name (e.g. 'ipc', 'curve')

    .. versionadded:: libzmq-4.1
    .. versionadded:: 14.1
    """
    _check_version((4,1), 'zmq.has')
    cdef bytes ccap
    if isinstance(capability, str):
        capability = capability.encode('utf8')
    ccap = capability
    return bool(zmq_has(ccap))

def curve_keypair():
    """generate a Z85 key pair for use with zmq.CURVE security

    Requires libzmq (≥ 4.0) to have been built with CURVE support.

    .. versionadded:: libzmq-4.0
    .. versionadded:: 14.0

    Returns
    -------
    (public, secret) : two bytestrings
        The public and private key pair as 40 byte z85-encoded bytestrings.
    """
    cdef int rc
    cdef char[64] public_key
    cdef char[64] secret_key
    _check_version((4,0), "curve_keypair")
    # see huge comment in libzmq/src/random.cpp
    # about threadsafety of random initialization
    rc = zmq_curve_keypair(public_key, secret_key)
    _check_rc(rc)
    return public_key, secret_key


def curve_public(secret_key):
    """ Compute the public key corresponding to a secret key for use
    with zmq.CURVE security

    Requires libzmq (≥ 4.2) to have been built with CURVE support.

    Parameters
    ----------
    private
        The private key as a 40 byte z85-encoded bytestring
    Returns
    -------
    bytestring
        The public key as a 40 byte z85-encoded bytestring.
    """
    if isinstance(secret_key, str):
        secret_key = secret_key.encode('utf8')
    if not len(secret_key) == 40:
        raise ValueError('secret key must be a 40 byte z85 encoded string')

    cdef int rc
    cdef char[64] public_key
    cdef char* c_secret_key = secret_key
    _check_version((4,2), "curve_public")
    # see huge comment in libzmq/src/random.cpp
    # about threadsafety of random initialization
    rc = zmq_curve_public(public_key, c_secret_key)
    _check_rc(rc)
    return public_key[:40]


__all__ = ['has', 'curve_keypair', 'curve_public']
