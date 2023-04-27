"""Future-returning APIs for coroutines."""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

from collections import namedtuple, deque
from itertools import chain

from zmq import EVENTS, POLLOUT, POLLIN
import zmq as _zmq

_FutureEvent = namedtuple('_FutureEvent', ('future', 'kind', 'kwargs', 'msg'))

# These are incomplete classes and need a Mixin for compatibility with an eventloop
# defining the followig attributes:
#
# _Future
# _READ
# _WRITE
# _default_loop()


class _AsyncPoller(_zmq.Poller):
    """Poller that returns a Future on poll, instead of blocking."""

    def poll(self, timeout=-1):
        """Return a Future for a poll event"""
        future = self._Future()
        if timeout == 0:
            try:
                result = super(_AsyncPoller, self).poll(0)
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(result)
            return future
        
        loop = self._default_loop()
        
        # register Future to be called as soon as any event is available on any socket
        watcher = self._Future()
        
        # watch raw sockets:
        raw_sockets = []
        def wake_raw(*args):
            if not watcher.done():
                watcher.set_result(None)

        watcher.add_done_callback(lambda f: self._unwatch_raw_sockets(loop, *raw_sockets))

        for socket, mask in self.sockets:
            if isinstance(socket, _zmq.Socket):
                if not isinstance(socket, self._socket_class):
                    # it's a blocking zmq.Socket, wrap it in async
                    socket = self._socket_class.from_socket(socket)
                if mask & _zmq.POLLIN:
                    socket._add_recv_event('poll', future=watcher)
                if mask & _zmq.POLLOUT:
                    socket._add_send_event('poll', future=watcher)
            else:
                raw_sockets.append(socket)
                evt = 0
                if mask & _zmq.POLLIN:
                    evt |= self._READ
                if mask & _zmq.POLLOUT:
                    evt |= self._WRITE
                self._watch_raw_socket(loop, socket, evt, wake_raw)

        def on_poll_ready(f):
            if future.done():
                return
            if watcher.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    # RuntimeError may be called during teardown
                    pass
                return
            if watcher.exception():
                future.set_exception(watcher.exception())
            else:
                try:
                    result = super(_AsyncPoller, self).poll(0)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
        watcher.add_done_callback(on_poll_ready)
        
        if timeout is not None and timeout > 0:
            # schedule cancel to fire on poll timeout, if any
            def trigger_timeout():
                if not watcher.done():
                    watcher.set_result(None)
            
            timeout_handle = loop.call_later(
                1e-3 * timeout,
                trigger_timeout
            )
            def cancel_timeout(f):
                if hasattr(timeout_handle, 'cancel'):
                    timeout_handle.cancel()
                else:
                    loop.remove_timeout(timeout_handle)
            future.add_done_callback(cancel_timeout)

        def cancel_watcher(f):
            if not watcher.done():
                watcher.cancel()
        future.add_done_callback(cancel_watcher)

        return future


class _AsyncSocket(_zmq.Socket):


    # Warning : these class variables are only here to allow to call super().__setattr__.
    # They be overridden at instance initialization and not shared in the whole class
    _recv_futures = None
    _send_futures = None
    _state = 0
    _shadow_sock = None
    _poller_class = _AsyncPoller
    io_loop = None
    _fd = None

    def __init__(self, context=None, socket_type=-1, io_loop=None, **kwargs):
        if isinstance(context, _zmq.Socket):
            context, from_socket = (None, context)
        else:
            from_socket = kwargs.pop('_from_socket', None)
        if from_socket is not None:
            super(_AsyncSocket, self).__init__(shadow=from_socket.underlying)
            self._shadow_sock = from_socket
        else:
            super(_AsyncSocket, self).__init__(context, socket_type, **kwargs)
            self._shadow_sock = _zmq.Socket.shadow(self.underlying)
        self.io_loop = io_loop or self._default_loop()
        self._recv_futures = deque()
        self._send_futures = deque()
        self._state = 0
        self._fd = self._shadow_sock.FD
        self._init_io_state()

    @classmethod
    def from_socket(cls, socket, io_loop=None):
        """Create an async socket from an existing Socket"""
        return cls(_from_socket=socket, io_loop=io_loop)

    def close(self, linger=None):
        if not self.closed:
            for event in list(chain(self._recv_futures, self._send_futures)):
                if not event.future.done():
                    try:
                        event.future.cancel()
                    except RuntimeError:
                        # RuntimeError may be called during teardown
                        pass
            self._clear_io_state()
        super(_AsyncSocket, self).close(linger=linger)
    close.__doc__ = _zmq.Socket.close.__doc__

    def get(self, key):
        result = super(_AsyncSocket, self).get(key)
        if key == EVENTS:
            self._schedule_remaining_events(result)
        return result
    get.__doc__ = _zmq.Socket.get.__doc__

    def recv_multipart(self, flags=0, copy=True, track=False):
        """Receive a complete multipart zmq message.
        
        Returns a Future whose result will be a multipart message.
        """
        return self._add_recv_event('recv_multipart',
            dict(flags=flags, copy=copy, track=track)
        )
    
    def recv(self, flags=0, copy=True, track=False):
        """Receive a single zmq frame.

        Returns a Future, whose result will be the received frame.

        Recommend using recv_multipart instead.
        """
        return self._add_recv_event('recv',
            dict(flags=flags, copy=copy, track=track)
        )

    def send_multipart(self, msg, flags=0, copy=True, track=False, **kwargs):
        """Send a complete multipart zmq message.

        Returns a Future that resolves when sending is complete.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        return self._add_send_event('send_multipart', msg=msg, kwargs=kwargs)

    def send(self, msg, flags=0, copy=True, track=False, **kwargs):
        """Send a single zmq frame.

        Returns a Future that resolves when sending is complete.

        Recommend using send_multipart instead.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        kwargs.update(dict(flags=flags, copy=copy, track=track))
        return self._add_send_event('send', msg=msg, kwargs=kwargs)

    def _deserialize(self, recvd, load):
        """Deserialize with Futures"""
        f = self._Future()
        def _chain(_):
            """Chain result through serialization to recvd"""
            if f.done():
                return
            if recvd.exception():
                f.set_exception(recvd.exception())
            else:
                buf = recvd.result()
                try:
                    loaded = load(buf)
                except Exception as e:
                    f.set_exception(e)
                else:
                    f.set_result(loaded)
        recvd.add_done_callback(_chain)

        def _chain_cancel(_):
            """Chain cancellation from f to recvd"""
            if recvd.done():
                return
            if f.cancelled():
                recvd.cancel()
        f.add_done_callback(_chain_cancel)

        return f

    def poll(self, timeout=None, flags=_zmq.POLLIN):
        """poll the socket for events

        returns a Future for the poll results.
        """

        if self.closed:
            raise _zmq.ZMQError(_zmq.ENOTSUP)

        p = self._poller_class()
        p.register(self, flags)
        f = p.poll(timeout)

        future = self._Future()
        def unwrap_result(f):
            if future.done():
                return
            if f.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    # RuntimeError may be called during teardown
                    pass
                return
            if f.exception():
                future.set_exception(f.exception())
            else:
                evts = dict(f.result())
                future.set_result(evts.get(self, 0))

        if f.done():
            # hook up result if
            unwrap_result(f)
        else:
            f.add_done_callback(unwrap_result)
        return future

    def _add_timeout(self, future, timeout):
        """Add a timeout for a send or recv Future"""
        def future_timeout():
            if future.done():
                # future already resolved, do nothing
                return

            # raise EAGAIN
            future.set_exception(_zmq.Again())
        self._call_later(timeout, future_timeout)

    def _call_later(self, delay, callback):
        """Schedule a function to be called later

        Override for different IOLoop implementations

        Tornado and asyncio happen to both have ioloop.call_later
        with the same signature.
        """
        self.io_loop.call_later(delay, callback)

    @staticmethod
    def _remove_finished_future(future, event_list):
        """Make sure that futures are removed from the event list when they resolve

        Avoids delaying cleanup until the next send/recv event,
        which may never come.
        """
        for f_idx, (f, kind, kwargs, _) in enumerate(event_list):
            if f is future:
                break
        else:
            return

        # "future" instance is shared between sockets, but each socket has its own event list.
        event_list.remove(event_list[f_idx])

    def _add_recv_event(self, kind, kwargs=None, future=None):
        """Add a recv event, returning the corresponding Future"""
        f = future or self._Future()
        if kind.startswith('recv') and kwargs.get('flags', 0) & _zmq.DONTWAIT:
            # short-circuit non-blocking calls
            recv = getattr(self._shadow_sock, kind)
            try:
                r = recv(**kwargs)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)
            return f

        # we add it to the list of futures before we add the timeout as the
        # timeout will remove the future from recv_futures to avoid leaks
        self._recv_futures.append(
            _FutureEvent(f, kind, kwargs, msg=None)
        )

        # Don't let the Future sit in _recv_events after it's done
        f.add_done_callback(lambda f: self._remove_finished_future(f, self._recv_futures))

        if hasattr(_zmq, 'RCVTIMEO'):
            timeout_ms = self._shadow_sock.rcvtimeo
            if timeout_ms >= 0:
                self._add_timeout(f, timeout_ms * 1e-3)

        if self._shadow_sock.get(EVENTS) & POLLIN:
            # recv immediately, if we can
            self._handle_recv()
        if self._recv_futures:
            self._add_io_state(POLLIN)
        return f

    def _add_send_event(self, kind, msg=None, kwargs=None, future=None):
        """Add a send event, returning the corresponding Future"""
        f = future or self._Future()
        # attempt send with DONTWAIT if no futures are waiting
        # short-circuit for sends that will resolve immediately
        # only call if no send Futures are waiting
        if (
            kind in ('send', 'send_multipart')
            and not self._send_futures
        ):
            flags = kwargs.get('flags', 0)
            nowait_kwargs = kwargs.copy()
            nowait_kwargs['flags'] = flags | _zmq.DONTWAIT

            # short-circuit non-blocking calls
            send = getattr(self._shadow_sock, kind)
            # track if the send resolved or not
            # (EAGAIN if DONTWAIT is not set should proceed with)
            finish_early = True
            try:
                r = send(msg, **nowait_kwargs)
            except _zmq.Again as e:
                if flags & _zmq.DONTWAIT:
                    f.set_exception(e)
                else:
                    # EAGAIN raised and DONTWAIT not requested,
                    # proceed with async send
                    finish_early = False
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)

            if finish_early:
                # short-circuit resolved, return finished Future
                # schedule wake for recv if there are any receivers waiting
                if self._recv_futures:
                    self._schedule_remaining_events()
                return f

        # we add it to the list of futures before we add the timeout as the
        # timeout will remove the future from recv_futures to avoid leaks
        self._send_futures.append(
            _FutureEvent(f, kind, kwargs=kwargs, msg=msg)
        )
        # Don't let the Future sit in _send_futures after it's done
        f.add_done_callback(lambda f: self._remove_finished_future(f, self._send_futures))

        if hasattr(_zmq, 'SNDTIMEO'):
            timeout_ms = self._shadow_sock.get(_zmq.SNDTIMEO)
            if timeout_ms >= 0:
                self._add_timeout(f, timeout_ms * 1e-3)

        self._add_io_state(POLLOUT)
        return f

    def _handle_recv(self):
        """Handle recv events"""
        if not self._shadow_sock.get(EVENTS) & POLLIN:
            # event triggered, but state may have been changed between trigger and callback
            return
        f = None
        while self._recv_futures:
            f, kind, kwargs, _ = self._recv_futures.popleft()
            # skip any cancelled futures
            if f.done():
                f = None
            else:
                break

        if not self._recv_futures:
            self._drop_io_state(POLLIN)

        if f is None:
            return

        if kind == 'poll':
            # on poll event, just signal ready, nothing else.
            f.set_result(None)
            return
        elif kind == 'recv_multipart':
            recv = self._shadow_sock.recv_multipart
        elif kind == 'recv':
            recv = self._shadow_sock.recv
        else:
            raise ValueError("Unhandled recv event type: %r" % kind)
        
        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = recv(**kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)
    
    def _handle_send(self):
        if not self._shadow_sock.get(EVENTS) & POLLOUT:
            # event triggered, but state may have been changed between trigger and callback
            return
        f = None
        while self._send_futures:
            f, kind, kwargs, msg = self._send_futures.popleft()
            # skip any cancelled futures
            if f.done():
                f = None
            else:
                break
        
        if not self._send_futures:
            self._drop_io_state(POLLOUT)

        if f is None:
            return
        
        if kind == 'poll':
            # on poll event, just signal ready, nothing else.
            f.set_result(None)
            return
        elif kind == 'send_multipart':
            send = self._shadow_sock.send_multipart
        elif kind == 'send':
            send = self._shadow_sock.send
        else:
            raise ValueError("Unhandled send event type: %r" % kind)
        
        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = send(msg, **kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)
    
    # event masking from ZMQStream
    def _handle_events(self, fd=0, events=0):
        """Dispatch IO events to _handle_recv, etc."""
        zmq_events = self._shadow_sock.get(EVENTS)
        if zmq_events & _zmq.POLLIN:
            self._handle_recv()
        if zmq_events & _zmq.POLLOUT:
            self._handle_send()
        self._schedule_remaining_events()

    def _schedule_remaining_events(self, events=None):
        """Schedule a call to handle_events next loop iteration

        If there are still events to handle.
        """
        # edge-triggered handling
        # allow passing events in, in case this is triggered by retrieving events,
        # so we don't have to retrieve it twice.
        if self._state == 0:
            # not watching for anything, nothing to schedule
            return
        if events is None:
            events = self._shadow_sock.get(EVENTS)
        if events & self._state:
            self._call_later(0, self._handle_events)

    def _add_io_state(self, state):
        """Add io_state to poller."""
        if self._state != state:
            state = self._state = self._state | state
        self._update_handler(self._state)

    def _drop_io_state(self, state):
        """Stop poller from watching an io_state."""
        if self._state & state:
            self._state = self._state & (~state)
        self._update_handler(self._state)

    def _update_handler(self, state):
        """Update IOLoop handler with state.

        zmq FD is always read-only.
        """
        self._schedule_remaining_events()

    def _init_io_state(self):
        """initialize the ioloop event handler"""
        self.io_loop.add_handler(self._shadow_sock, self._handle_events, self._READ)
        self._call_later(0, self._handle_events)

    def _clear_io_state(self):
        """unregister the ioloop event handler

        called once during close
        """
        fd = self._shadow_sock
        if self._shadow_sock.closed:
            fd = self._fd
        self.io_loop.remove_handler(fd)


