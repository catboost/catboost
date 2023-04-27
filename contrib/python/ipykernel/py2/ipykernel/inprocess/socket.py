""" Defines a dummy socket implementing (part of) the zmq.Socket interface. """

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import abc
import warnings
try:
    from queue import Queue  # Py 3
except ImportError:
    from Queue import Queue  # Py 2

import zmq

from traitlets import HasTraits, Instance, Int
from ipython_genutils.py3compat import with_metaclass

#-----------------------------------------------------------------------------
# Generic socket interface
#-----------------------------------------------------------------------------

class SocketABC(with_metaclass(abc.ABCMeta, object)):
    
    @abc.abstractmethod
    def recv_multipart(self, flags=0, copy=True, track=False):
        raise NotImplementedError

    @abc.abstractmethod
    def send_multipart(self, msg_parts, flags=0, copy=True, track=False):
        raise NotImplementedError
    
    @classmethod
    def register(cls, other_cls):
        if other_cls is not DummySocket:
            warnings.warn("SocketABC is deprecated since ipykernel version 4.5.0.",
                    DeprecationWarning, stacklevel=2)
        abc.ABCMeta.register(cls, other_cls)

#-----------------------------------------------------------------------------
# Dummy socket class
#-----------------------------------------------------------------------------

class DummySocket(HasTraits):
    """ A dummy socket implementing (part of) the zmq.Socket interface. """

    queue = Instance(Queue, ())
    message_sent = Int(0) # Should be an Event
    context = Instance(zmq.Context)
    def _context_default(self):
        return zmq.Context.instance()

    #-------------------------------------------------------------------------
    # Socket interface
    #-------------------------------------------------------------------------

    def recv_multipart(self, flags=0, copy=True, track=False):
        return self.queue.get_nowait()

    def send_multipart(self, msg_parts, flags=0, copy=True, track=False):
        msg_parts = list(map(zmq.Message, msg_parts))
        self.queue.put_nowait(msg_parts)
        self.message_sent += 1

SocketABC.register(DummySocket)
