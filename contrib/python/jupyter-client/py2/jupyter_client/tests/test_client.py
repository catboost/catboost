"""Tests for the KernelClient"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.


import os
pjoin = os.path.join
from unittest import TestCase

from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel, NATIVE_KERNEL_NAME
from jupyter_client.manager import start_new_kernel
from .utils import test_env

import pytest

from ipython_genutils.py3compat import string_types
from IPython.utils.capture import capture_output

TIMEOUT = 30

class TestKernelClient(TestCase):
    def setUp(self):
        self.env_patch = test_env()
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        try:
            KernelSpecManager().get_kernel_spec(NATIVE_KERNEL_NAME)
        except NoSuchKernel:
            pytest.skip()
        self.km, self.kc = start_new_kernel(kernel_name=NATIVE_KERNEL_NAME)
        self.addCleanup(self.kc.stop_channels)
        self.addCleanup(self.km.shutdown_kernel)

    def test_execute_interactive(self):
        kc = self.kc

        with capture_output() as io:
            reply = kc.execute_interactive("print('hello')", timeout=TIMEOUT)
        assert 'hello' in io.stdout
        assert reply['content']['status'] == 'ok'
    
    def _check_reply(self, reply_type, reply):
        self.assertIsInstance(reply, dict)
        self.assertEqual(reply['header']['msg_type'], reply_type + '_reply')
        self.assertEqual(reply['parent_header']['msg_type'], reply_type + '_request')

    def test_history(self):
        kc = self.kc
        msg_id = kc.history(session=0)
        self.assertIsInstance(msg_id, string_types)
        reply = kc.history(session=0, reply=True, timeout=TIMEOUT)
        self._check_reply('history', reply)

    def test_inspect(self):
        kc = self.kc
        msg_id = kc.inspect('who cares')
        self.assertIsInstance(msg_id, string_types)
        reply = kc.inspect('code', reply=True, timeout=TIMEOUT)
        self._check_reply('inspect', reply)

    def test_complete(self):
        kc = self.kc
        msg_id = kc.complete('who cares')
        self.assertIsInstance(msg_id, string_types)
        reply = kc.complete('code', reply=True, timeout=TIMEOUT)
        self._check_reply('complete', reply)

    def test_kernel_info(self):
        kc = self.kc
        msg_id = kc.kernel_info()
        self.assertIsInstance(msg_id, string_types)
        reply = kc.kernel_info(reply=True, timeout=TIMEOUT)
        self._check_reply('kernel_info', reply)

    def test_comm_info(self):
        kc = self.kc
        msg_id = kc.comm_info()
        self.assertIsInstance(msg_id, string_types)
        reply = kc.comm_info(reply=True, timeout=TIMEOUT)
        self._check_reply('comm_info', reply)

    def test_shutdown(self):
        kc = self.kc
        msg_id = kc.shutdown()
        self.assertIsInstance(msg_id, string_types)
        reply = kc.shutdown(reply=True, timeout=TIMEOUT)
        self._check_reply('shutdown', reply)
