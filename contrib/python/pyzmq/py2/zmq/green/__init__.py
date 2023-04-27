# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (C) 2011-2012 Travis Cline
#
#  This file is part of pyzmq
#  It is adapted from upstream project zeromq_gevent under the New BSD License
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------

"""zmq.green - gevent compatibility with zeromq.

Usage
-----

Instead of importing zmq directly, do so in the following manner:

..

    import zmq.green as zmq


Any calls that would have blocked the current thread will now only block the
current green thread.

This compatibility is accomplished by ensuring the nonblocking flag is set
before any blocking operation and the ØMQ file descriptor is polled internally
to trigger needed events.
"""

from zmq import *
from zmq.green.core import _Context, _Socket
from zmq.green.poll import _Poller
Context = _Context
Socket = _Socket
Poller = _Poller

from zmq.green.device import device

