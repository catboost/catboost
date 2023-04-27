"""Declare basic string types unambiguously for various Python versions.

Authors
-------
* MinRK
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import sys

if sys.version_info[0] >= 3:
    bytes = bytes
    unicode = str
    basestring = (bytes, unicode)
else:
    unicode = unicode
    bytes = str
    basestring = basestring

def cast_bytes(s, encoding='utf8', errors='strict'):
    """cast unicode or bytes to bytes"""
    if isinstance(s, bytes):
        return s
    elif isinstance(s, unicode):
        return s.encode(encoding, errors)
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)

def cast_unicode(s, encoding='utf8', errors='strict'):
    """cast bytes or unicode to unicode"""
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, unicode):
        return s
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)

# give short 'b' alias for cast_bytes, so that we can use fake b('stuff')
# to simulate b'stuff'
b = asbytes = cast_bytes
u = cast_unicode

__all__ = ['asbytes', 'bytes', 'unicode', 'basestring', 'b', 'u', 'cast_bytes', 'cast_unicode']
