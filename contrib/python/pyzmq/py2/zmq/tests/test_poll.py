# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import os
import sys
import time

from pytest import mark

import zmq

from zmq.tests import PollZMQTestCase, have_gevent, GreenTest

def wait():
    time.sleep(.25)


class TestPoll(PollZMQTestCase):

    Poller = zmq.Poller

    def test_pair(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)

        # Sleep to allow sockets to connect.
        wait()

        poller = self.Poller()
        poller.register(s1, zmq.POLLIN|zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN|zmq.POLLOUT)
        # Poll result should contain both sockets
        socks = dict(poller.poll())
        # Now make sure that both are send ready.
        self.assertEqual(socks[s1], zmq.POLLOUT)
        self.assertEqual(socks[s2], zmq.POLLOUT)
        # Now do a send on both, wait and test for zmq.POLLOUT|zmq.POLLIN
        s1.send(b'msg1')
        s2.send(b'msg2')
        wait()
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLOUT|zmq.POLLIN)
        self.assertEqual(socks[s2], zmq.POLLOUT|zmq.POLLIN)
        # Make sure that both are in POLLOUT after recv.
        s1.recv()
        s2.recv()
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLOUT)
        self.assertEqual(socks[s2], zmq.POLLOUT)

        poller.unregister(s1)
        poller.unregister(s2)


    def test_reqrep(self):
        s1, s2 = self.create_bound_pair(zmq.REP, zmq.REQ)

        # Sleep to allow sockets to connect.
        wait()

        poller = self.Poller()
        poller.register(s1, zmq.POLLIN|zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN|zmq.POLLOUT)

        # Make sure that s1 is in state 0 and s2 is in POLLOUT
        socks = dict(poller.poll())
        self.assertEqual(s1 in socks, 0)
        self.assertEqual(socks[s2], zmq.POLLOUT)

        # Make sure that s2 goes immediately into state 0 after send.
        s2.send(b'msg1')
        socks = dict(poller.poll())
        self.assertEqual(s2 in socks, 0)

        # Make sure that s1 goes into POLLIN state after a time.sleep().
        time.sleep(0.5)
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLIN)

        # Make sure that s1 goes into POLLOUT after recv.
        s1.recv()
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLOUT)

        # Make sure s1 goes into state 0 after send.
        s1.send(b'msg2')
        socks = dict(poller.poll())
        self.assertEqual(s1 in socks, 0)

        # Wait and then see that s2 is in POLLIN.
        time.sleep(0.5)
        socks = dict(poller.poll())
        self.assertEqual(socks[s2], zmq.POLLIN)

        # Make sure that s2 is in POLLOUT after recv.
        s2.recv()
        socks = dict(poller.poll())
        self.assertEqual(socks[s2], zmq.POLLOUT)

        poller.unregister(s1)
        poller.unregister(s2)

    def test_no_events(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN|zmq.POLLOUT)
        poller.register(s2, 0)
        self.assertTrue(s1 in poller)
        self.assertFalse(s2 in poller)
        poller.register(s1, 0)
        self.assertFalse(s1 in poller)

    def test_pubsub(self):
        s1, s2 = self.create_bound_pair(zmq.PUB, zmq.SUB)
        s2.setsockopt(zmq.SUBSCRIBE, b'')

        # Sleep to allow sockets to connect.
        wait()

        poller = self.Poller()
        poller.register(s1, zmq.POLLIN|zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN)

        # Now make sure that both are send ready.
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLOUT)
        self.assertEqual(s2 in socks, 0)
        # Make sure that s1 stays in POLLOUT after a send.
        s1.send(b'msg1')
        socks = dict(poller.poll())
        self.assertEqual(socks[s1], zmq.POLLOUT)

        # Make sure that s2 is POLLIN after waiting.
        wait()
        socks = dict(poller.poll())
        self.assertEqual(socks[s2], zmq.POLLIN)

        # Make sure that s2 goes into 0 after recv.
        s2.recv()
        socks = dict(poller.poll())
        self.assertEqual(s2 in socks, 0)

        poller.unregister(s1)
        poller.unregister(s2)

    @mark.skipif(sys.platform.startswith('win'), reason='Windows')
    def test_raw(self):
        r, w = os.pipe()
        r = os.fdopen(r, 'rb')
        w = os.fdopen(w, 'wb')
        p = self.Poller()
        p.register(r, zmq.POLLIN)
        socks = dict(p.poll(1))
        assert socks == {}
        w.write(b'x')
        w.flush()
        socks = dict(p.poll(1))
        assert socks == {r.fileno(): zmq.POLLIN}
        w.close()
        r.close()

    def test_timeout(self):
        """make sure Poller.poll timeout has the right units (milliseconds)."""
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN)
        tic = time.time()
        evt = poller.poll(.005)
        toc = time.time()
        self.assertTrue(toc-tic < 0.1)
        tic = time.time()
        evt = poller.poll(5)
        toc = time.time()
        self.assertTrue(toc-tic < 0.1)
        self.assertTrue(toc-tic > .001)
        tic = time.time()
        evt = poller.poll(500)
        toc = time.time()
        self.assertTrue(toc-tic < 1)
        self.assertTrue(toc-tic > 0.1)

class TestSelect(PollZMQTestCase):

    def test_pair(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)

        # Sleep to allow sockets to connect.
        wait()

        rlist, wlist, xlist = zmq.select([s1, s2], [s1, s2], [s1, s2])
        self.assert_(s1 in wlist)
        self.assert_(s2 in wlist)
        self.assert_(s1 not in rlist)
        self.assert_(s2 not in rlist)

    def test_timeout(self):
        """make sure select timeout has the right units (seconds)."""
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        tic = time.time()
        r,w,x = zmq.select([s1,s2],[],[],.005)
        toc = time.time()
        self.assertTrue(toc-tic < 1)
        self.assertTrue(toc-tic > 0.001)
        tic = time.time()
        r,w,x = zmq.select([s1,s2],[],[],.25)
        toc = time.time()
        self.assertTrue(toc-tic < 1)
        self.assertTrue(toc-tic > 0.1)


if have_gevent:
    import gevent
    from zmq import green as gzmq

    class TestPollGreen(GreenTest, TestPoll):
        Poller = gzmq.Poller

        def test_wakeup(self):
            s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
            poller = self.Poller()
            poller.register(s2, zmq.POLLIN)

            tic = time.time()
            r = gevent.spawn(lambda: poller.poll(10000))
            s = gevent.spawn(lambda: s1.send(b'msg1'))
            r.join()
            toc = time.time()
            self.assertTrue(toc-tic < 1)
        
        def test_socket_poll(self):
            s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)

            tic = time.time()
            r = gevent.spawn(lambda: s2.poll(10000))
            s = gevent.spawn(lambda: s1.send(b'msg1'))
            r.join()
            toc = time.time()
            self.assertTrue(toc-tic < 1)

