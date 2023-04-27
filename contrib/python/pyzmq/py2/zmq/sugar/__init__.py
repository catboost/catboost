"""pure-Python sugar wrappers for core 0MQ objects."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from zmq.sugar import (
    constants, context, frame, poll, socket, tracker, version
)
from zmq import error

__all__ = ['constants']
for submod in (
    constants, context, error, frame, poll, socket, tracker, version
):
    __all__.extend(submod.__all__)

from zmq.error import *
from zmq.sugar.context import *
from zmq.sugar.tracker import *
from zmq.sugar.socket import *
from zmq.sugar.constants import *
from zmq.sugar.frame import *
from zmq.sugar.poll import *
from zmq.sugar.version import *

# deprecated:
from zmq.sugar.stopwatch import Stopwatch
__all__.append('Stopwatch')
