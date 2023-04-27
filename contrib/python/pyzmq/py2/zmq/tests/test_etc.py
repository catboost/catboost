# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import sys

import zmq

from pytest import mark

@mark.skipif('zmq.zmq_version_info() < (4,1)')
def test_has():
    assert not zmq.has('something weird')
    has_ipc = zmq.has('ipc')
    not_windows = not sys.platform.startswith('win')
    assert has_ipc == not_windows

@mark.skipif(not hasattr(zmq, '_libzmq'), reason="bundled libzmq")
def test_has_curve():
    """bundled libzmq has curve support"""
    assert zmq.has('curve')
