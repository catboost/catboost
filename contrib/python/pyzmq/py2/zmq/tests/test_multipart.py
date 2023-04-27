# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import zmq


from zmq.tests import BaseZMQTestCase, SkipTest, have_gevent, GreenTest


class TestMultipart(BaseZMQTestCase):

    def test_router_dealer(self):
        router, dealer = self.create_bound_pair(zmq.ROUTER, zmq.DEALER)

        msg1 = b'message1'
        dealer.send(msg1)
        ident = self.recv(router)
        more = router.rcvmore
        self.assertEqual(more, True)
        msg2 = self.recv(router)
        self.assertEqual(msg1, msg2)
        more = router.rcvmore
        self.assertEqual(more, False)
    
    def test_basic_multipart(self):
        a,b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        msg = [ b'hi', b'there', b'b']
        a.send_multipart(msg)
        recvd = b.recv_multipart()
        self.assertEqual(msg, recvd)

if have_gevent:
    class TestMultipartGreen(GreenTest, TestMultipart):
        pass
