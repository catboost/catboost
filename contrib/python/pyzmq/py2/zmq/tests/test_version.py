# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from unittest import TestCase
import zmq
from zmq.sugar import version


class TestVersion(TestCase):

    def test_pyzmq_version(self):
        vs = zmq.pyzmq_version()
        vs2 = zmq.__version__
        self.assertTrue(isinstance(vs, str))
        if zmq.__revision__:
            self.assertEqual(vs, '@'.join(vs2, zmq.__revision__))
        else:
            self.assertEqual(vs, vs2)
        if version.VERSION_EXTRA:
            self.assertTrue(version.VERSION_EXTRA in vs)
            self.assertTrue(version.VERSION_EXTRA in vs2)

    def test_pyzmq_version_info(self):
        info = zmq.pyzmq_version_info()
        self.assertTrue(isinstance(info, tuple))
        for n in info[:3]:
            self.assertTrue(isinstance(n, int))
        if version.VERSION_EXTRA:
            self.assertEqual(len(info), 4)
            self.assertEqual(info[-1], float('inf'))
        else:
            self.assertEqual(len(info), 3)

    def test_zmq_version_info(self):
        info = zmq.zmq_version_info()
        self.assertTrue(isinstance(info, tuple))
        for n in info[:3]:
            self.assertTrue(isinstance(n, int))

    def test_zmq_version(self):
        v = zmq.zmq_version()
        self.assertTrue(isinstance(v, str))

