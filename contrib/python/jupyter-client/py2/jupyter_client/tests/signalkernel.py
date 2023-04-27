"""Test kernel for signalling subprocesses"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import print_function

from subprocess import Popen, PIPE
import sys
import time

from ipykernel.displayhook import ZMQDisplayHook
from ipykernel.kernelbase import Kernel
from ipykernel.kernelapp import IPKernelApp


class SignalTestKernel(Kernel):
    """Kernel for testing subprocess signaling"""
    implementation = 'signaltest'
    implementation_version = '0.0'
    banner = ''

    def __init__(self, **kwargs):
        kwargs.pop('user_ns', None)
        super(SignalTestKernel, self).__init__(**kwargs)
        self.children = []

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):
        code = code.strip()
        reply = {
            'status': 'ok',
            'user_expressions': {},
        }
        if code == 'start':
            child = Popen(['bash', '-i', '-c', 'sleep 30'], stderr=PIPE)
            self.children.append(child)
            reply['user_expressions']['pid'] = self.children[-1].pid
        elif code == 'check':
            reply['user_expressions']['poll'] = [ child.poll() for child in self.children ]
        elif code == 'sleep':
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                reply['user_expressions']['interrupted'] = True
            else:
                reply['user_expressions']['interrupted'] = False
        else:
            reply['status'] = 'error'
            reply['ename'] = 'Error'
            reply['evalue'] = code
            reply['traceback'] = ['no such command: %s' % code]
        return reply
    
    def kernel_info_request(self, *args, **kwargs):
        """Add delay to kernel_info_request
        
        triggers slow-response code in KernelClient.wait_for_ready
        """
        return super(SignalTestKernel, self).kernel_info_request(*args, **kwargs)

class SignalTestApp(IPKernelApp):
    kernel_class = SignalTestKernel
    def init_io(self):
        # Overridden to disable stdout/stderr capture
        self.displayhook = ZMQDisplayHook(self.session, self.iopub_socket)

if __name__ == '__main__':
    # make startup artificially slow,
    # so that we exercise client logic for slow-starting kernels
    time.sleep(2)
    SignalTestApp.launch_instance()
