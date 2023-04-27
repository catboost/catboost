from zmq.eventloop.zmqstream import *

from zmq.green.eventloop.ioloop import IOLoop

RealZMQStream = ZMQStream

class ZMQStream(RealZMQStream):
    
    def __init__(self, socket, io_loop=None):
        io_loop = io_loop or IOLoop.instance()
        super(ZMQStream, self).__init__(socket, io_loop=io_loop)
