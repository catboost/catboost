"""ZAP Authenticator integrated with the tornado IOLoop.

.. versionadded:: 14.1
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from tornado import ioloop
from zmq.eventloop import zmqstream
from .base import Authenticator


class IOLoopAuthenticator(Authenticator):
    """ZAP authentication for use in the tornado IOLoop"""

    def __init__(self, context=None, encoding='utf-8', log=None, io_loop=None):
        super(IOLoopAuthenticator, self).__init__(context, encoding, log)
        self.zap_stream = None
        self.io_loop = io_loop or ioloop.IOLoop.current()

    def start(self):
        """Start ZAP authentication"""
        super(IOLoopAuthenticator, self).start()
        self.zap_stream = zmqstream.ZMQStream(self.zap_socket, self.io_loop)
        self.zap_stream.on_recv(self.handle_zap_message)

    def stop(self):
        """Stop ZAP authentication"""
        if self.zap_stream:
            self.zap_stream.close()
            self.zap_stream = None
        super(IOLoopAuthenticator, self).stop()

__all__ = ['IOLoopAuthenticator']
