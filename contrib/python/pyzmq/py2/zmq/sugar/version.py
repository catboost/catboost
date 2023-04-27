"""PyZMQ and 0MQ version functions."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from zmq.backend import zmq_version_info


VERSION_MAJOR = 19
VERSION_MINOR = 0
VERSION_PATCH = 2
VERSION_EXTRA = ""
__version__ = '%i.%i.%i' % (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

if VERSION_EXTRA:
    __version__ = "%s.%s" % (__version__, VERSION_EXTRA)
    version_info = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, float('inf'))
else:
    version_info = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

__revision__ = ''

def pyzmq_version():
    """return the version of pyzmq as a string"""
    if __revision__:
        return '@'.join([__version__,__revision__[:6]])
    else:
        return __version__

def pyzmq_version_info():
    """return the pyzmq version as a tuple of at least three numbers
    
    If pyzmq is a development version, `inf` will be appended after the third integer.
    """
    return version_info


def zmq_version():
    """return the version of libzmq as a string"""
    return "%i.%i.%i" % zmq_version_info()


__all__ = ['zmq_version', 'zmq_version_info',
           'pyzmq_version','pyzmq_version_info',
           '__version__', '__revision__'
]

