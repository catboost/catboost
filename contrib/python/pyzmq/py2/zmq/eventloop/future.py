"""Future-returning APIs for tornado coroutines.

.. seealso::

    :mod:`zmq.asyncio`

"""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import zmq as _zmq

from zmq._future import _AsyncPoller, _AsyncSocket

from tornado.concurrent import Future
from tornado.ioloop import IOLoop

class CancelledError(Exception):
    pass

class _TornadoFuture(Future):
    """Subclass Tornado Future, reinstating cancellation."""
    def cancel(self):
        if self.done():
            return False
        self.set_exception(CancelledError())
        return True
    
    def cancelled(self):
        return self.done() and isinstance(self.exception(), CancelledError)

# mixin for tornado/asyncio compatibility

class _AsyncTornado(object):
    _Future = _TornadoFuture
    _READ = IOLoop.READ
    _WRITE = IOLoop.WRITE
    def _default_loop(self):
        return IOLoop.current()


class Poller(_AsyncTornado, _AsyncPoller):
    def _watch_raw_socket(self, loop, socket, evt, f):
        """Schedule callback for a raw socket"""
        loop.add_handler(socket, lambda *args: f(), evt)

    def _unwatch_raw_sockets(self, loop, *sockets):
        """Unschedule callback for a raw socket"""
        for socket in sockets:
            loop.remove_handler(socket)


class Socket(_AsyncTornado, _AsyncSocket):
    _poller_class = Poller

Poller._socket_class = Socket

class Context(_zmq.Context):

    # avoid sharing instance with base Context class
    _instance = None

    io_loop = None
    @staticmethod
    def _socket_class(self, socket_type):
        return Socket(self, socket_type, io_loop=self.io_loop)

    def __init__(self, *args, **kwargs):
        io_loop = kwargs.pop('io_loop', None)
        super(Context, self).__init__(*args, **kwargs)
        self.io_loop = io_loop or IOLoop.current()

