# -*- coding: utf8 -*-
# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import copy
import errno
import json
import os
import platform
import time
import warnings
import socket
import sys
try:
    from unittest import mock
except ImportError:
    mock = None

import pytest
from pytest import mark

import zmq
from zmq.tests import (
    BaseZMQTestCase, SkipTest, have_gevent, GreenTest, skip_pypy
)
from zmq.utils.strtypes import unicode

pypy = platform.python_implementation().lower() == 'pypy'
windows = platform.platform().lower().startswith('windows')
on_travis = bool(os.environ.get('TRAVIS_PYTHON_VERSION'))

# polling on windows is slow
POLL_TIMEOUT = 1000 if windows else 100

class TestSocket(BaseZMQTestCase):

    def test_create(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        # Superluminal protocol not yet implemented
        self.assertRaisesErrno(zmq.EPROTONOSUPPORT, s.bind, 'ftl://a')
        self.assertRaisesErrno(zmq.EPROTONOSUPPORT, s.connect, 'ftl://a')
        self.assertRaisesErrno(zmq.EINVAL, s.bind, 'tcp://')
        s.close()
        del ctx
    
    def test_context_manager(self):
        url = 'inproc://a'
        with self.Context() as ctx:
            with ctx.socket(zmq.PUSH) as a:
                a.bind(url)
                with ctx.socket(zmq.PULL) as b:
                    b.connect(url)
                    msg = b'hi'
                    a.send(msg)
                    rcvd = self.recv(b)
                    self.assertEqual(rcvd, msg)
                self.assertEqual(b.closed, True)
            self.assertEqual(a.closed, True)
        self.assertEqual(ctx.closed, True)

    def test_dir(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        self.assertTrue('send' in dir(s))
        self.assertTrue('IDENTITY' in dir(s))
        self.assertTrue('AFFINITY' in dir(s))
        self.assertTrue('FD' in dir(s))
        s.close()
        ctx.term()

    @mark.skipif(mock is None, reason="requires unittest.mock")
    def test_mockable(self):
        s = self.socket(zmq.SUB)
        m = mock.Mock(spec=s)
        s.close()

    def test_bind_unicode(self):
        s = self.socket(zmq.PUB)
        p = s.bind_to_random_port(unicode("tcp://*"))

    def test_connect_unicode(self):
        s = self.socket(zmq.PUB)
        s.connect(unicode("tcp://127.0.0.1:5555"))

    def test_bind_to_random_port(self):
        # Check that bind_to_random_port do not hide useful exception
        ctx = self.Context()
        c = ctx.socket(zmq.PUB)
        # Invalid format
        try:
            c.bind_to_random_port('tcp:*')
        except zmq.ZMQError as e:
            self.assertEqual(e.errno, zmq.EINVAL)
        # Invalid protocol
        try:
            c.bind_to_random_port('rand://*')
        except zmq.ZMQError as e:
            self.assertEqual(e.errno, zmq.EPROTONOSUPPORT)

    def test_identity(self):
        s = self.context.socket(zmq.PULL)
        self.sockets.append(s)
        ident = b'identity\0\0'
        s.identity = ident
        self.assertEqual(s.get(zmq.IDENTITY), ident)

    def test_unicode_sockopts(self):
        """test setting/getting sockopts with unicode strings"""
        topic = "tést"
        if str is not unicode:
            topic = topic.decode('utf8')
        p,s = self.create_bound_pair(zmq.PUB, zmq.SUB)
        self.assertEqual(s.send_unicode, s.send_unicode)
        self.assertEqual(p.recv_unicode, p.recv_unicode)
        self.assertRaises(TypeError, s.setsockopt, zmq.SUBSCRIBE, topic)
        self.assertRaises(TypeError, s.setsockopt, zmq.IDENTITY, topic)
        s.setsockopt_unicode(zmq.IDENTITY, topic, 'utf16')
        self.assertRaises(TypeError, s.setsockopt, zmq.AFFINITY, topic)
        s.setsockopt_unicode(zmq.SUBSCRIBE, topic)
        self.assertRaises(TypeError, s.getsockopt_unicode, zmq.AFFINITY)
        self.assertRaisesErrno(zmq.EINVAL, s.getsockopt_unicode, zmq.SUBSCRIBE)
        
        identb = s.getsockopt(zmq.IDENTITY)
        identu = identb.decode('utf16')
        identu2 = s.getsockopt_unicode(zmq.IDENTITY, 'utf16')
        self.assertEqual(identu, identu2)
        time.sleep(0.1) # wait for connection/subscription
        p.send_unicode(topic,zmq.SNDMORE)
        p.send_unicode(topic*2, encoding='latin-1')
        self.assertEqual(topic, s.recv_unicode())
        self.assertEqual(topic*2, s.recv_unicode(encoding='latin-1'))
    
    def test_int_sockopts(self):
        "test integer sockopts"
        v = zmq.zmq_version_info()
        if v < (3,0):
            default_hwm = 0
        else:
            default_hwm = 1000
        p,s = self.create_bound_pair(zmq.PUB, zmq.SUB)
        p.setsockopt(zmq.LINGER, 0)
        self.assertEqual(p.getsockopt(zmq.LINGER), 0)
        p.setsockopt(zmq.LINGER, -1)
        self.assertEqual(p.getsockopt(zmq.LINGER), -1)
        self.assertEqual(p.hwm, default_hwm)
        p.hwm = 11
        self.assertEqual(p.hwm, 11)
        # p.setsockopt(zmq.EVENTS, zmq.POLLIN)
        self.assertEqual(p.getsockopt(zmq.EVENTS), zmq.POLLOUT)
        self.assertRaisesErrno(zmq.EINVAL, p.setsockopt,zmq.EVENTS, 2**7-1)
        self.assertEqual(p.getsockopt(zmq.TYPE), p.socket_type)
        self.assertEqual(p.getsockopt(zmq.TYPE), zmq.PUB)
        self.assertEqual(s.getsockopt(zmq.TYPE), s.socket_type)
        self.assertEqual(s.getsockopt(zmq.TYPE), zmq.SUB)
        
        # check for overflow / wrong type:
        errors = []
        backref = {}
        constants = zmq.constants
        for name in constants.__all__:
            value = getattr(constants, name)
            if isinstance(value, int):
                backref[value] = name
        for opt in zmq.constants.int_sockopts.union(zmq.constants.int64_sockopts):
            sopt = backref[opt]
            if sopt.startswith((
                'ROUTER', 'XPUB', 'TCP', 'FAIL',
                'REQ_', 'CURVE_', 'PROBE_ROUTER',
                'IPC_FILTER', 'GSSAPI', 'STREAM_',
                'VMCI_BUFFER_SIZE', 'VMCI_BUFFER_MIN_SIZE',
                'VMCI_BUFFER_MAX_SIZE', 'VMCI_CONNECT_TIMEOUT',
                )):
                # some sockopts are write-only
                continue
            try:
                n = p.getsockopt(opt)
            except zmq.ZMQError as e:
                errors.append("getsockopt(zmq.%s) raised '%s'."%(sopt, e))
            else:
                if n > 2**31:
                    errors.append("getsockopt(zmq.%s) returned a ridiculous value."
                                    " It is probably the wrong type."%sopt)
        if errors:
            self.fail('\n'.join([''] + errors))
    
    def test_bad_sockopts(self):
        """Test that appropriate errors are raised on bad socket options"""
        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        s.setsockopt(zmq.LINGER, 0)
        # unrecognized int sockopts pass through to libzmq, and should raise EINVAL
        self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, 9999, 5)
        self.assertRaisesErrno(zmq.EINVAL, s.getsockopt, 9999)
        # but only int sockopts are allowed through this way, otherwise raise a TypeError
        self.assertRaises(TypeError, s.setsockopt, 9999, b"5")
        # some sockopts are valid in general, but not on every socket:
        self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, zmq.SUBSCRIBE, b'hi')
    
    def test_sockopt_roundtrip(self):
        "test set/getsockopt roundtrip."
        p = self.context.socket(zmq.PUB)
        self.sockets.append(p)
        p.setsockopt(zmq.LINGER, 11)
        self.assertEqual(p.getsockopt(zmq.LINGER), 11)
    
    def test_send_unicode(self):
        "test sending unicode objects"
        a,b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        self.sockets.extend([a,b])
        u = "çπ§"
        if str is not unicode:
            u = u.decode('utf8')
        self.assertRaises(TypeError, a.send, u,copy=False)
        self.assertRaises(TypeError, a.send, u,copy=True)
        a.send_unicode(u)
        s = b.recv()
        self.assertEqual(s,u.encode('utf8'))
        self.assertEqual(s.decode('utf8'),u)
        a.send_unicode(u,encoding='utf16')
        s = b.recv_unicode(encoding='utf16')
        self.assertEqual(s,u)
    
    def test_send_multipart_check_type(self):
        "check type on all frames in send_multipart"
        a,b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        self.sockets.extend([a,b])
        self.assertRaises(TypeError, a.send_multipart, [b'a', 5])
        a.send_multipart([b'b'])
        rcvd = self.recv_multipart(b)
        self.assertEqual(rcvd, [b'b'])
    
    @skip_pypy
    def test_tracker(self):
        "test the MessageTracker object for tracking when zmq is done with a buffer"
        addr = 'tcp://127.0.0.1'
        # get a port:
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        iface = "%s:%i" % (addr, port)
        sock.close()
        time.sleep(0.1)

        a = self.context.socket(zmq.PUSH)
        b = self.context.socket(zmq.PULL)
        self.sockets.extend([a,b])
        a.connect(iface)
        time.sleep(0.1)
        p1 = a.send(b'something', copy=False, track=True)
        assert isinstance(p1, zmq.MessageTracker)
        assert p1 is zmq._FINISHED_TRACKER
        # small message, should start done
        assert p1.done

        # disable zero-copy threshold
        a.copy_threshold = 0

        p2 = a.send_multipart([b'something', b'else'], copy=False, track=True)
        assert isinstance(p2, zmq.MessageTracker)
        assert not p2.done

        b.bind(iface)
        msg = self.recv_multipart(b)
        for i in range(10):
            if p1.done:
                break
            time.sleep(0.1)
        self.assertEqual(p1.done, True)
        self.assertEqual(msg, [b'something'])
        msg = self.recv_multipart(b)
        for i in range(10):
            if p2.done:
                break
            time.sleep(0.1)
        self.assertEqual(p2.done, True)
        self.assertEqual(msg, [b'something', b'else'])
        m = zmq.Frame(b"again", copy=False, track=True)
        self.assertEqual(m.tracker.done, False)
        p1 = a.send(m, copy=False)
        p2 = a.send(m, copy=False)
        self.assertEqual(m.tracker.done, False)
        self.assertEqual(p1.done, False)
        self.assertEqual(p2.done, False)
        msg = self.recv_multipart(b)
        self.assertEqual(m.tracker.done, False)
        self.assertEqual(msg, [b'again'])
        msg = self.recv_multipart(b)
        self.assertEqual(m.tracker.done, False)
        self.assertEqual(msg, [b'again'])
        self.assertEqual(p1.done, False)
        self.assertEqual(p2.done, False)
        pm = m.tracker
        del m
        for i in range(10):
            if p1.done:
                break
            time.sleep(0.1)
        self.assertEqual(p1.done, True)
        self.assertEqual(p2.done, True)
        m = zmq.Frame(b'something', track=False)
        self.assertRaises(ValueError, a.send, m, copy=False, track=True)

    def test_close(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        s.close()
        self.assertRaisesErrno(zmq.ENOTSOCK, s.bind, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.connect, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.setsockopt, zmq.SUBSCRIBE, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.send, b'asdf')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.recv)
        del ctx
    
    def test_attr(self):
        """set setting/getting sockopts as attributes"""
        s = self.context.socket(zmq.DEALER)
        self.sockets.append(s)
        linger = 10
        s.linger = linger
        self.assertEqual(linger, s.linger)
        self.assertEqual(linger, s.getsockopt(zmq.LINGER))
        self.assertEqual(s.fd, s.getsockopt(zmq.FD))
    
    def test_bad_attr(self):
        s = self.context.socket(zmq.DEALER)
        self.sockets.append(s)
        try:
            s.apple='foo'
        except AttributeError:
            pass
        else:
            self.fail("bad setattr should have raised AttributeError")
        try:
            s.apple
        except AttributeError:
            pass
        else:
            self.fail("bad getattr should have raised AttributeError")

    def test_subclass(self):
        """subclasses can assign attributes"""
        class S(zmq.Socket):
            a = None
            def __init__(self, *a, **kw):
                self.a=-1
                super(S, self).__init__(*a, **kw)
        
        s = S(self.context, zmq.REP)
        self.sockets.append(s)
        self.assertEqual(s.a, -1)
        s.a=1
        self.assertEqual(s.a, 1)
        a=s.a
        self.assertEqual(a, 1)
    
    def test_recv_multipart(self):
        a,b = self.create_bound_pair()
        msg = b'hi'
        for i in range(3):
            a.send(msg)
        time.sleep(0.1)
        for i in range(3):
            self.assertEqual(self.recv_multipart(b), [msg])
    
    def test_close_after_destroy(self):
        """s.close() after ctx.destroy() should be fine"""
        ctx = self.Context()
        s = ctx.socket(zmq.REP)
        ctx.destroy()
        # reaper is not instantaneous
        time.sleep(1e-2)
        s.close()
        self.assertTrue(s.closed)
    
    def test_poll(self):
        a,b = self.create_bound_pair()
        tic = time.time()
        evt = a.poll(POLL_TIMEOUT)
        self.assertEqual(evt, 0)
        evt = a.poll(POLL_TIMEOUT, zmq.POLLOUT)
        self.assertEqual(evt, zmq.POLLOUT)
        msg = b'hi'
        a.send(msg)
        evt = b.poll(POLL_TIMEOUT)
        self.assertEqual(evt, zmq.POLLIN)
        msg2 = self.recv(b)
        evt = b.poll(POLL_TIMEOUT)
        self.assertEqual(evt, 0)
        self.assertEqual(msg2, msg)
    
    def test_ipc_path_max_length(self):
        """IPC_PATH_MAX_LEN is a sensible value"""
        if zmq.IPC_PATH_MAX_LEN == 0:
            raise SkipTest("IPC_PATH_MAX_LEN undefined")
        
        msg = "Surprising value for IPC_PATH_MAX_LEN: %s" % zmq.IPC_PATH_MAX_LEN
        self.assertTrue(zmq.IPC_PATH_MAX_LEN > 30, msg)
        self.assertTrue(zmq.IPC_PATH_MAX_LEN < 1025, msg)

    def test_ipc_path_max_length_msg(self):
        if zmq.IPC_PATH_MAX_LEN == 0:
            raise SkipTest("IPC_PATH_MAX_LEN undefined")
        
        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        try:
            s.bind('ipc://{0}'.format('a' * (zmq.IPC_PATH_MAX_LEN + 1)))
        except zmq.ZMQError as e:
            self.assertTrue(str(zmq.IPC_PATH_MAX_LEN) in e.strerror)

    @mark.skipif(windows, reason="ipc not supported on Windows.")
    def test_ipc_path_no_such_file_or_directory_message(self):
        """Display the ipc path in case of an ENOENT exception"""
        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        invalid_path = '/foo/bar'
        with pytest.raises(zmq.ZMQError) as error:
            s.bind('ipc://{0}'.format(invalid_path))
        assert error.value.errno == errno.ENOENT
        error_message = str(error.value)
        assert invalid_path in error_message
        assert "no such file or directory" in error_message.lower()

    def test_hwm(self):
        zmq3 = zmq.zmq_version_info()[0] >= 3
        for stype in (zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER):
            s = self.context.socket(stype)
            s.hwm = 100
            self.assertEqual(s.hwm, 100)
            if zmq3:
                try:
                    self.assertEqual(s.sndhwm, 100)
                except AttributeError:
                    pass
                try:
                    self.assertEqual(s.rcvhwm, 100)
                except AttributeError:
                    pass
            s.close()

    def test_copy(self):
        s = self.socket(zmq.PUB)
        scopy = copy.copy(s)
        sdcopy = copy.deepcopy(s)
        self.assert_(scopy._shadow)
        self.assert_(sdcopy._shadow)
        self.assertEqual(s.underlying, scopy.underlying)
        self.assertEqual(s.underlying, sdcopy.underlying)
        s.close()

    def test_send_buffer(self):
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        for buffer_type in (memoryview, bytearray):
            rawbytes = str(buffer_type).encode('ascii')
            msg = buffer_type(rawbytes)
            a.send(msg)
            recvd = b.recv()
            assert recvd == rawbytes

    def test_shadow(self):
        p = self.socket(zmq.PUSH)
        p.bind("tcp://127.0.0.1:5555")
        p2 = zmq.Socket.shadow(p.underlying)
        self.assertEqual(p.underlying, p2.underlying)
        s = self.socket(zmq.PULL)
        s2 = zmq.Socket.shadow(s.underlying)
        self.assertNotEqual(s.underlying, p.underlying)
        self.assertEqual(s.underlying, s2.underlying)
        s2.connect("tcp://127.0.0.1:5555")
        sent = b'hi'
        p2.send(sent)
        rcvd = self.recv(s2)
        self.assertEqual(rcvd, sent)

    def test_shadow_pyczmq(self):
        try:
            from pyczmq import zctx, zsocket
        except Exception:
            raise SkipTest("Requires pyczmq")

        ctx = zctx.new()
        ca = zsocket.new(ctx, zmq.PUSH)
        cb = zsocket.new(ctx, zmq.PULL)
        a = zmq.Socket.shadow(ca)
        b = zmq.Socket.shadow(cb)
        a.bind("inproc://a")
        b.connect("inproc://a")
        a.send(b'hi')
        rcvd = self.recv(b)
        self.assertEqual(rcvd, b'hi')

    def test_subscribe_method(self):
        pub, sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
        sub.subscribe('prefix')
        sub.subscribe = 'c'
        p = zmq.Poller()
        p.register(sub, zmq.POLLIN)
        # wait for subscription handshake
        for i in range(100):
            pub.send(b'canary')
            events = p.poll(250)
            if events:
                break
        self.recv(sub)
        pub.send(b'prefixmessage')
        msg = self.recv(sub)
        self.assertEqual(msg, b'prefixmessage')
        sub.unsubscribe('prefix')
        pub.send(b'prefixmessage')
        events = p.poll(1000)
        self.assertEqual(events, [])

    # Travis can't handle how much memory PyPy uses on this test
    @mark.skipif(
        (
            pypy and on_travis
        ) or (
            sys.maxsize < 2**32
        ) or (
            windows
        ),
        reason="only run on 64b and not on Travis."
    )
    @mark.large
    def test_large_send(self):
        c = os.urandom(1)
        N = 2**31 + 1
        try:
            buf = c * N
        except MemoryError as e:
            raise SkipTest("Not enough memory: %s" % e)
        a, b = self.create_bound_pair()
        try:
            a.send(buf, copy=False)
            rcvd = b.recv(copy=False)
        except MemoryError as e:
            raise SkipTest("Not enough memory: %s" % e)
        # sample the front and back of the received message
        # without checking the whole content
        # Python 2: items in memoryview are bytes
        # Python 3: items im memoryview are int
        byte = c if sys.version_info < (3,) else ord(c)
        view = memoryview(rcvd)
        assert len(view) == N
        assert view[0] == byte
        assert view[-1] == byte

    def test_custom_serialize(self):
        a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)
        def serialize(msg):
            frames = []
            frames.extend(msg.get('identities', []))
            content = json.dumps(msg['content']).encode('utf8')
            frames.append(content)
            return frames

        def deserialize(frames):
            identities = frames[:-1]
            content = json.loads(frames[-1].decode('utf8'))
            return {
                'identities': identities,
                'content': content,
            }
        
        msg = {
            'content': {
                'a': 5,
                'b': 'bee',
            }
        }
        a.send_serialized(msg, serialize)
        recvd = b.recv_serialized(deserialize)
        assert recvd['content'] == msg['content']
        assert recvd['identities']
        # bounce back, tests identities
        b.send_serialized(recvd, serialize)
        r2 = a.recv_serialized(deserialize)
        assert r2['content'] == msg['content']
        assert not r2['identities']


if have_gevent and not windows:
    import gevent
    
    class TestSocketGreen(GreenTest, TestSocket):
        test_bad_attr = GreenTest.skip_green
        test_close_after_destroy = GreenTest.skip_green
        
        def test_timeout(self):
            a,b = self.create_bound_pair()
            g = gevent.spawn_later(0.5, lambda: a.send(b'hi'))
            timeout = gevent.Timeout(0.1)
            timeout.start()
            self.assertRaises(gevent.Timeout, b.recv)
            g.kill()
        
        @mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason="requires RCVTIMEO")
        def test_warn_set_timeo(self):
            s = self.context.socket(zmq.REQ)
            with warnings.catch_warnings(record=True) as w:
                s.rcvtimeo = 5
            s.close()
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            

        @mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason="requires SNDTIMEO")
        def test_warn_get_timeo(self):
            s = self.context.socket(zmq.REQ)
            with warnings.catch_warnings(record=True) as w:
                s.sndtimeo
            s.close()
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
