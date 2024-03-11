"""A kernel manager with a tornado IOLoop"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import absolute_import

from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream

from traitlets import (
    Instance,
    Type,
)

from jupyter_client.manager import KernelManager
from .restarter import IOLoopKernelRestarter


def as_zmqstream(f):
    def wrapped(self, *args, **kwargs):
        socket = f(self, *args, **kwargs)
        return ZMQStream(socket, self.loop)
    return wrapped

class IOLoopKernelManager(KernelManager):

    loop = Instance('tornado.ioloop.IOLoop')
    def _loop_default(self):
        return ioloop.IOLoop.current()

    restarter_class = Type(
        default_value=IOLoopKernelRestarter,
        klass=IOLoopKernelRestarter,
        help=(
            'Type of KernelRestarter to use. '
            'Must be a subclass of IOLoopKernelRestarter.\n'
            'Override this to customize how kernel restarts are managed.'
        ),
        config=True,
    )
    _restarter = Instance('jupyter_client.ioloop.IOLoopKernelRestarter', allow_none=True)

    def start_restarter(self):
        if self.autorestart and self.has_kernel:
            if self._restarter is None:
                self._restarter = self.restarter_class(
                    kernel_manager=self, loop=self.loop,
                    parent=self, log=self.log
                )
            self._restarter.start()

    def stop_restarter(self):
        if self.autorestart:
            if self._restarter is not None:
                self._restarter.stop()

    connect_shell = as_zmqstream(KernelManager.connect_shell)
    connect_control = as_zmqstream(KernelManager.connect_control)
    connect_iopub = as_zmqstream(KernelManager.connect_iopub)
    connect_stdin = as_zmqstream(KernelManager.connect_stdin)
    connect_hb = as_zmqstream(KernelManager.connect_hb)
