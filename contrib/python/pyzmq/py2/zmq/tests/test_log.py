# encoding: utf-8

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import logging
import time
from unittest import TestCase

import zmq
from zmq.log import handlers
from zmq.utils.strtypes import b, u
from zmq.tests import BaseZMQTestCase


class TestPubLog(BaseZMQTestCase):
    
    iface = 'inproc://zmqlog'
    topic= 'zmq'
    
    @property
    def logger(self):
        # print dir(self)
        logger = logging.getLogger('zmqtest')
        logger.setLevel(logging.DEBUG)
        return logger
    
    def connect_handler(self, topic=None):
        topic = self.topic if topic is None else topic
        logger = self.logger
        pub,sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
        handler = handlers.PUBHandler(pub)
        handler.setLevel(logging.DEBUG)
        handler.root_topic = topic
        logger.addHandler(handler)
        sub.setsockopt(zmq.SUBSCRIBE, b(topic))
        time.sleep(0.1)
        return logger, handler, sub
    
    def test_init_iface(self):
        logger = self.logger
        ctx = self.context
        handler = handlers.PUBHandler(self.iface)
        self.assertFalse(handler.ctx is ctx)
        self.sockets.append(handler.socket)
        # handler.ctx.term()
        handler = handlers.PUBHandler(self.iface, self.context)
        self.sockets.append(handler.socket)
        self.assertTrue(handler.ctx is ctx)
        handler.setLevel(logging.DEBUG)
        handler.root_topic = self.topic
        logger.addHandler(handler)
        sub = ctx.socket(zmq.SUB)
        self.sockets.append(sub)
        sub.setsockopt(zmq.SUBSCRIBE, b(self.topic))
        sub.connect(self.iface)
        import time; time.sleep(0.25)
        msg1 = 'message'
        logger.info(msg1)
        
        (topic, msg2) = sub.recv_multipart()
        self.assertEqual(topic, b'zmq.INFO')
        self.assertEqual(msg2, b(msg1)+b'\n')
        logger.removeHandler(handler)
    
    def test_init_socket(self):
        pub,sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
        logger = self.logger
        handler = handlers.PUBHandler(pub)
        handler.setLevel(logging.DEBUG)
        handler.root_topic = self.topic
        logger.addHandler(handler)
        
        self.assertTrue(handler.socket is pub)
        self.assertTrue(handler.ctx is pub.context)
        self.assertTrue(handler.ctx is self.context)
        sub.setsockopt(zmq.SUBSCRIBE, b(self.topic))
        import time; time.sleep(0.1)
        msg1 = 'message'
        logger.info(msg1)
        
        (topic, msg2) = sub.recv_multipart()
        self.assertEqual(topic, b'zmq.INFO')
        self.assertEqual(msg2, b(msg1)+b'\n')
        logger.removeHandler(handler)
    
    def test_root_topic(self):
        logger, handler, sub = self.connect_handler()
        handler.socket.bind(self.iface)
        sub2 = sub.context.socket(zmq.SUB)
        self.sockets.append(sub2)
        sub2.connect(self.iface)
        sub2.setsockopt(zmq.SUBSCRIBE, b'')
        handler.root_topic = b'twoonly'
        msg1 = 'ignored'
        logger.info(msg1)
        self.assertRaisesErrno(zmq.EAGAIN, sub.recv, zmq.NOBLOCK)
        topic,msg2 = sub2.recv_multipart()
        self.assertEqual(topic, b'twoonly.INFO')
        self.assertEqual(msg2, b(msg1)+b'\n')
        
        logger.removeHandler(handler)
    
    def test_blank_root_topic(self):
        logger, handler, sub_everything = self.connect_handler()
        sub_everything.setsockopt(zmq.SUBSCRIBE, b'')
        handler.socket.bind(self.iface)
        sub_only_info = sub_everything.context.socket(zmq.SUB)
        self.sockets.append(sub_only_info)
        sub_only_info.connect(self.iface)
        sub_only_info.setsockopt(zmq.SUBSCRIBE, b'INFO')
        handler.setRootTopic(b'')
        msg_debug = 'debug_message'
        logger.debug(msg_debug)
        self.assertRaisesErrno(zmq.EAGAIN, sub_only_info.recv, zmq.NOBLOCK)
        topic, msg_debug_response = sub_everything.recv_multipart()
        self.assertEqual(topic, b'DEBUG')
        msg_info = 'info_message'
        logger.info(msg_info)
        topic, msg_info_response_everything = sub_everything.recv_multipart()
        self.assertEqual(topic, b'INFO')
        topic, msg_info_response_onlyinfo = sub_only_info.recv_multipart()
        self.assertEqual(topic, b'INFO')
        self.assertEqual(msg_info_response_everything, msg_info_response_onlyinfo)

        logger.removeHandler(handler)

    def test_unicode_message(self):
        logger, handler, sub = self.connect_handler()
        base_topic = b(self.topic + '.INFO')
        for msg, expected in [
            (u('hello'), [base_topic, b('hello\n')]),
            (u('héllo'), [base_topic, b('héllo\n')]),
            (u('tøpic::héllo'), [base_topic + b('.tøpic'), b('héllo\n')]),
        ]:
            logger.info(msg)
            received = sub.recv_multipart()
            self.assertEqual(received, expected)
        logger.removeHandler(handler)

    def test_set_info_formatter_via_property(self):
        logger, handler, sub = self.connect_handler()
        handler.formatters[logging.INFO] = logging.Formatter("%(message)s UNITTEST\n")
        handler.socket.bind(self.iface)
        sub.setsockopt(zmq.SUBSCRIBE, b(handler.root_topic))
        logger.info('info message')
        topic, msg = sub.recv_multipart()
        self.assertEqual(msg, b'info message UNITTEST\n')
        logger.removeHandler(handler)

    def test_custom_global_formatter(self):
        logger, handler, sub = self.connect_handler()
        formatter = logging.Formatter("UNITTEST %(message)s")
        handler.setFormatter(formatter)
        handler.socket.bind(self.iface)
        sub.setsockopt(zmq.SUBSCRIBE, b(handler.root_topic))
        logger.info('info message')
        topic, msg = sub.recv_multipart()
        self.assertEqual(msg, b'UNITTEST info message')
        logger.debug('debug message')
        topic, msg = sub.recv_multipart()
        self.assertEqual(msg, b'UNITTEST debug message')
        logger.removeHandler(handler)

    def test_custom_debug_formatter(self):
        logger, handler, sub = self.connect_handler()
        formatter = logging.Formatter("UNITTEST DEBUG %(message)s")
        handler.setFormatter(formatter, logging.DEBUG)
        handler.socket.bind(self.iface)
        sub.setsockopt(zmq.SUBSCRIBE, b(handler.root_topic))
        logger.info('info message')
        topic, msg = sub.recv_multipart()
        self.assertEqual(msg, b'info message\n')
        logger.debug('debug message')
        topic, msg = sub.recv_multipart()
        self.assertEqual(msg, b'UNITTEST DEBUG debug message')
        logger.removeHandler(handler)
