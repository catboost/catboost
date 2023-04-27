"""Tracker for zero-copy messages with 0MQ."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import time

try:
    # below 3.3
    from threading import _Event as Event
except (ImportError, AttributeError):
    # python throws ImportError, cython throws AttributeError
    from threading import Event

from zmq.error import NotDone
from zmq.backend import Frame

class MessageTracker(object):
    """MessageTracker(*towatch)

    A class for tracking if 0MQ is done using one or more messages.

    When you send a 0MQ message, it is not sent immediately. The 0MQ IO thread
    sends the message at some later time. Often you want to know when 0MQ has
    actually sent the message though. This is complicated by the fact that
    a single 0MQ message can be sent multiple times using different sockets.
    This class allows you to track all of the 0MQ usages of a message.

    Parameters
    ----------
    towatch : Event, MessageTracker, Message instances.
        This objects to track. This class can track the low-level
        Events used by the Message class, other MessageTrackers or
        actual Messages.
    """
    events = None
    peers = None

    def __init__(self, *towatch):
        """MessageTracker(*towatch)

        Create a message tracker to track a set of mesages.

        Parameters
        ----------
        *towatch : tuple of Event, MessageTracker, Message instances.
            This list of objects to track. This class can track the low-level
            Events used by the Message class, other MessageTrackers or 
            actual Messages.
        """
        self.events = set()
        self.peers = set()
        for obj in towatch:
            if isinstance(obj, Event):
                self.events.add(obj)
            elif isinstance(obj, MessageTracker):
                self.peers.add(obj)
            elif isinstance(obj, Frame):
                if not obj.tracker:
                    raise ValueError("Not a tracked message")
                self.peers.add(obj.tracker)
            else:
                raise TypeError("Require Events or Message Frames, not %s"%type(obj))
    
    @property
    def done(self):
        """Is 0MQ completely done with the message(s) being tracked?"""
        for evt in self.events:
            if not evt.is_set():
                return False
        for pm in self.peers:
            if not pm.done:
                return False
        return True
    
    def wait(self, timeout=-1):
        """mt.wait(timeout=-1)

        Wait for 0MQ to be done with the message or until `timeout`.

        Parameters
        ----------
        timeout : float [default: -1, wait forever]
            Maximum time in (s) to wait before raising NotDone.

        Returns
        -------
        None
            if done before `timeout`
        
        Raises
        ------
        NotDone
            if `timeout` reached before I am done.
        """
        tic = time.time()
        if timeout is False or timeout < 0:
            remaining = 3600*24*7 # a week
        else:
            remaining = timeout
        done = False
        for evt in self.events:
            if remaining < 0:
                raise NotDone
            evt.wait(timeout=remaining)
            if not evt.is_set():
                raise NotDone
            toc = time.time()
            remaining -= (toc-tic)
            tic = toc
        
        for peer in self.peers:
            if remaining < 0:
                raise NotDone
            peer.wait(timeout=remaining)
            toc = time.time()
            remaining -= (toc-tic)
            tic = toc

_FINISHED_TRACKER = MessageTracker()

__all__ = ['MessageTracker', '_FINISHED_TRACKER']
