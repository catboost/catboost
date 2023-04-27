"""A basic in process kernel monitor with autorestarting.

This watches a kernel's state using KernelManager.is_alive and auto
restarts the kernel if it dies.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import absolute_import
import warnings

from zmq.eventloop import ioloop

from jupyter_client.restarter import KernelRestarter
from traitlets import (
    Instance,
)

class IOLoopKernelRestarter(KernelRestarter):
    """Monitor and autorestart a kernel."""

    loop = Instance('tornado.ioloop.IOLoop')
    def _loop_default(self):
        warnings.warn("IOLoopKernelRestarter.loop is deprecated in jupyter-client 5.2",
            DeprecationWarning, stacklevel=4,
        )
        return ioloop.IOLoop.current()

    _pcallback = None

    def start(self):
        """Start the polling of the kernel."""
        if self._pcallback is None:
            self._pcallback = ioloop.PeriodicCallback(
                self.poll, 1000*self.time_to_dead,
            )
            self._pcallback.start()

    def stop(self):
        """Stop the kernel polling."""
        if self._pcallback is not None:
            self._pcallback.stop()
            self._pcallback = None
