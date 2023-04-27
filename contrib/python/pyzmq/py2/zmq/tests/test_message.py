# -*- coding: utf8 -*-
# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import copy
import sys
try:
    from sys import getrefcount as grc
except ImportError:
    grc = None

import time
from pprint import pprint
from unittest import TestCase

import zmq
from zmq.tests import BaseZMQTestCase, SkipTest, skip_pypy, PYPY
from zmq.utils.strtypes import unicode, bytes, b, u


# some useful constants:

x = b'x'

if grc:
    rc0 = grc(x)
    v = memoryview(x)
    view_rc = grc(x) - rc0

def await_gc(obj, rc):
    """wait for refcount on an object to drop to an expected value
    
    Necessary because of the zero-copy gc thread,
    which can take some time to receive its DECREF message.
    """
    for i in range(50):
        # rc + 2 because of the refs in this function
        if grc(obj) <= rc + 2:
            return
        time.sleep(0.05)
    
class TestFrame(BaseZMQTestCase):

    @skip_pypy
    def test_above_30(self):
        """Message above 30 bytes are never copied by 0MQ."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i)*x
            self.assertEqual(grc(s), 2)
            m = zmq.Frame(s, copy=False)
            self.assertEqual(grc(s), 4)
            del m
            await_gc(s, 2)
            self.assertEqual(grc(s), 2)
            del s

    def test_str(self):
        """Test the str representations of the Frames."""
        for i in range(16):
            s = (2**i)*x
            m = zmq.Frame(s)
            m_str = str(m)
            m_str_b = b(m_str) # py3compat
            self.assertEqual(s, m_str_b)

    def test_bytes(self):
        """Test the Frame.bytes property."""
        for i in range(1,16):
            s = (2**i)*x
            m = zmq.Frame(s)
            b = m.bytes
            self.assertEqual(s, m.bytes)
            if not PYPY:
                # check that it copies
                self.assert_(b is not s)
            # check that it copies only once
            self.assert_(b is m.bytes)

    def test_unicode(self):
        """Test the unicode representations of the Frames."""
        s = u('asdf')
        self.assertRaises(TypeError, zmq.Frame, s)
        for i in range(16):
            s = (2**i)*u('§')
            m = zmq.Frame(s.encode('utf8'))
            self.assertEqual(s, unicode(m.bytes,'utf8'))

    def test_len(self):
        """Test the len of the Frames."""
        for i in range(16):
            s = (2**i)*x
            m = zmq.Frame(s)
            self.assertEqual(len(s), len(m))

    @skip_pypy
    def test_lifecycle1(self):
        """Run through a ref counting cycle with a copy."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i)*x
            rc = 2
            self.assertEqual(grc(s), rc)
            m = zmq.Frame(s, copy=False)
            rc += 2
            self.assertEqual(grc(s), rc)
            m2 = copy.copy(m)
            rc += 1
            self.assertEqual(grc(s), rc)
            buf = m2.buffer

            rc += view_rc
            self.assertEqual(grc(s), rc)

            self.assertEqual(s, b(str(m)))
            self.assertEqual(s, bytes(m2))
            self.assertEqual(s, m.bytes)
            # self.assert_(s is str(m))
            # self.assert_(s is str(m2))
            del m2
            rc -= 1
            self.assertEqual(grc(s), rc)
            rc -= view_rc
            del buf
            self.assertEqual(grc(s), rc)
            del m
            rc -= 2
            await_gc(s, rc)
            self.assertEqual(grc(s), rc)
            self.assertEqual(rc, 2)
            del s

    @skip_pypy
    def test_lifecycle2(self):
        """Run through a different ref counting cycle with a copy."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i)*x
            rc = 2
            self.assertEqual(grc(s), rc)
            m = zmq.Frame(s, copy=False)
            rc += 2
            self.assertEqual(grc(s), rc)
            m2 = copy.copy(m)
            rc += 1
            self.assertEqual(grc(s), rc)
            buf = m.buffer
            rc += view_rc
            self.assertEqual(grc(s), rc)
            self.assertEqual(s, b(str(m)))
            self.assertEqual(s, bytes(m2))
            self.assertEqual(s, m2.bytes)
            self.assertEqual(s, m.bytes)
            # self.assert_(s is str(m))
            # self.assert_(s is str(m2))
            del buf
            self.assertEqual(grc(s), rc)
            del m
            # m.buffer is kept until m is del'd
            rc -= view_rc
            rc -= 1
            self.assertEqual(grc(s), rc)
            del m2
            rc -= 2
            await_gc(s, rc)
            self.assertEqual(grc(s), rc)
            self.assertEqual(rc, 2)
            del s
    
    @skip_pypy
    def test_tracker(self):
        m = zmq.Frame(b'asdf', copy=False, track=True)
        self.assertFalse(m.tracker.done)
        pm = zmq.MessageTracker(m)
        self.assertFalse(pm.done)
        del m
        for i in range(10):
            if pm.done:
                break
            time.sleep(0.1)
        self.assertTrue(pm.done)
    
    def test_no_tracker(self):
        m = zmq.Frame(b'asdf', track=False)
        self.assertEqual(m.tracker, None)
        m2 = copy.copy(m)
        self.assertEqual(m2.tracker, None)
        self.assertRaises(ValueError, zmq.MessageTracker, m)
    
    @skip_pypy
    def test_multi_tracker(self):
        m = zmq.Frame(b'asdf', copy=False, track=True)
        m2 = zmq.Frame(b'whoda', copy=False, track=True)
        mt = zmq.MessageTracker(m,m2)
        self.assertFalse(m.tracker.done)
        self.assertFalse(mt.done)
        self.assertRaises(zmq.NotDone, mt.wait, 0.1)
        del m
        time.sleep(0.1)
        self.assertRaises(zmq.NotDone, mt.wait, 0.1)
        self.assertFalse(mt.done)
        del m2
        self.assertTrue(mt.wait() is None)
        self.assertTrue(mt.done)
        
    
    def test_buffer_in(self):
        """test using a buffer as input"""
        ins = b("§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√")
        m = zmq.Frame(memoryview(ins))
    
    def test_bad_buffer_in(self):
        """test using a bad object"""
        self.assertRaises(TypeError, zmq.Frame, 5)
        self.assertRaises(TypeError, zmq.Frame, object())
        
    def test_buffer_out(self):
        """receiving buffered output"""
        ins = b("§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√")
        m = zmq.Frame(ins)
        outb = m.buffer
        self.assertTrue(isinstance(outb, memoryview))
        self.assert_(outb is m.buffer)
        self.assert_(m.buffer is m.buffer)
    
    @skip_pypy
    def test_memoryview_shape(self):
        """memoryview shape info"""
        if sys.version_info < (3,):
            raise SkipTest("only test memoryviews on Python 3")
        data = b("§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√")
        n = len(data)
        f = zmq.Frame(data)
        view1 = f.buffer
        self.assertEqual(view1.ndim, 1)
        self.assertEqual(view1.shape, (n,))
        self.assertEqual(view1.tobytes(), data)
        view2 = memoryview(f)
        self.assertEqual(view2.ndim, 1)
        self.assertEqual(view2.shape, (n,))
        self.assertEqual(view2.tobytes(), data)
    
    def test_multisend(self):
        """ensure that a message remains intact after multiple sends"""
        a,b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        s = b"message"
        m = zmq.Frame(s)
        self.assertEqual(s, m.bytes)
        
        a.send(m, copy=False)
        time.sleep(0.1)
        self.assertEqual(s, m.bytes)
        a.send(m, copy=False)
        time.sleep(0.1)
        self.assertEqual(s, m.bytes)
        a.send(m, copy=True)
        time.sleep(0.1)
        self.assertEqual(s, m.bytes)
        a.send(m, copy=True)
        time.sleep(0.1)
        self.assertEqual(s, m.bytes)
        for i in range(4):
            r = b.recv()
            self.assertEqual(s,r)
        self.assertEqual(s, m.bytes)

    def test_memoryview(self):
        """test messages from memoryview"""
        major,minor = sys.version_info[:2]
        if not (major >= 3 or (major == 2 and minor >= 7)):
            raise SkipTest("memoryviews only in python >= 2.7")

        s = b'carrotjuice'
        v = memoryview(s)
        m = zmq.Frame(s)
        buf = m.buffer
        s2 = buf.tobytes()
        self.assertEqual(s2,s)
        self.assertEqual(m.bytes,s)

    def test_noncopying_recv(self):
        """check for clobbering message buffers"""
        null = b'\0'*64
        sa,sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        for i in range(32):
            # try a few times
            sb.send(null, copy=False)
            m = sa.recv(copy=False)
            mb = m.bytes
            # buf = memoryview(m)
            buf = m.buffer
            del m
            for i in range(5):
                ff=b'\xff'*(40 + i*10)
                sb.send(ff, copy=False)
                m2 = sa.recv(copy=False)
                b = buf.tobytes()
                self.assertEqual(b, null)
                self.assertEqual(mb, null)
                self.assertEqual(m2.bytes, ff)

    @skip_pypy
    def test_buffer_numpy(self):
        """test non-copying numpy array messages"""
        try:
            import numpy
            from numpy.testing import assert_array_equal
        except ImportError:
            raise SkipTest("requires numpy")
        if sys.version_info < (2,7):
            raise SkipTest("requires new-style buffer interface (py >= 2.7)")
        rand = numpy.random.randint
        shapes = [ rand(2,5) for i in range(5) ]
        a,b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        dtypes = [int, float, '>i4', 'B']
        for i in range(1,len(shapes)+1):
            shape = shapes[:i]
            for dt in dtypes:
                A = numpy.empty(shape, dtype=dt)
                a.send(A, copy=False)
                msg = b.recv(copy=False)

                B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
                assert_array_equal(A, B)

            A = numpy.empty(shape, dtype=[('a', int), ('b', float), ('c', 'a32')])
            A['a'] = 1024
            A['b'] = 1e9
            A['c'] = 'hello there'
            a.send(A, copy=False)
            msg = b.recv(copy=False)

            B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
            assert_array_equal(A, B)

    def test_frame_more(self):
        """test Frame.more attribute"""
        frame = zmq.Frame(b"hello")
        self.assertFalse(frame.more)
        sa,sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        sa.send_multipart([b'hi', b'there'])
        frame = self.recv(sb, copy=False)
        self.assertTrue(frame.more)
        if zmq.zmq_version_info()[0] >= 3 and not PYPY:
            self.assertTrue(frame.get(zmq.MORE))
        frame = self.recv(sb, copy=False)
        self.assertFalse(frame.more)
        if zmq.zmq_version_info()[0] >= 3 and not PYPY:
            self.assertFalse(frame.get(zmq.MORE))

