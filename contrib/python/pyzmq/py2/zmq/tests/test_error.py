# -*- coding: utf8 -*-
# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import sys
import time
from threading import Thread

import zmq
from zmq import ZMQError, strerror, Again, ContextTerminated
from zmq.tests import BaseZMQTestCase

if sys.version_info[0] >= 3:
    long = int

class TestZMQError(BaseZMQTestCase):
    
    def test_strerror(self):
        """test that strerror gets the right type."""
        for i in range(10):
            e = strerror(i)
            self.assertTrue(isinstance(e, str))
    
    def test_zmqerror(self):
        for errno in range(10):
            e = ZMQError(errno)
            self.assertEqual(e.errno, errno)
            self.assertEqual(str(e), strerror(errno))
    
    def test_again(self):
        s = self.context.socket(zmq.REP)
        self.assertRaises(Again, s.recv, zmq.NOBLOCK)
        self.assertRaisesErrno(zmq.EAGAIN, s.recv, zmq.NOBLOCK)
        s.close()
    
    def atest_ctxterm(self):
        s = self.context.socket(zmq.REP)
        t = Thread(target=self.context.term)
        t.start()
        self.assertRaises(ContextTerminated, s.recv, zmq.NOBLOCK)
        self.assertRaisesErrno(zmq.TERM, s.recv, zmq.NOBLOCK)
        s.close()
        t.join()
