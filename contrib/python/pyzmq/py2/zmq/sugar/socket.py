# coding: utf-8
"""0MQ Socket pure Python methods."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import errno
import random
import sys
import warnings

import zmq
from zmq.backend import Socket as SocketBase
from .poll import Poller
from . import constants
from .attrsettr import AttributeSetter
from zmq.error import ZMQError, ZMQBindError
from zmq.utils import jsonapi
from zmq.utils.strtypes import bytes, unicode, basestring


from .constants import (
    SNDMORE, ENOTSUP, POLLIN,
    int64_sockopt_names,
    int_sockopt_names,
    bytes_sockopt_names,
    fd_sockopt_names,
)
try:
    import cPickle
    pickle = cPickle
except:
    cPickle = None
    import pickle

try:
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
except AttributeError:
    DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL


class Socket(SocketBase, AttributeSetter):
    """The ZMQ socket object

    To create a Socket, first create a Context::

        ctx = zmq.Context.instance()

    then call ``ctx.socket(socket_type)``::

        s = ctx.socket(zmq.ROUTER)

    """
    _shadow = False
    _monitor_socket = None

    def __init__(self, *a, **kw):
        super(Socket, self).__init__(*a, **kw)
        if 'shadow' in kw:
            self._shadow = True
        else:
            self._shadow = False

    def __del__(self):
        if not self._shadow:
            self.close()

    # socket as context manager:
    def __enter__(self):
        """Sockets are context managers
        
        .. versionadded:: 14.4
        """
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()
    
    #-------------------------------------------------------------------------
    # Socket creation
    #-------------------------------------------------------------------------

    def __copy__(self, memo=None):
        """Copying a Socket creates a shadow copy"""
        return self.__class__.shadow(self.underlying)

    __deepcopy__ = __copy__

    @classmethod
    def shadow(cls, address):
        """Shadow an existing libzmq socket

        address is the integer address of the libzmq socket
        or an FFI pointer to it.

        .. versionadded:: 14.1
        """
        from zmq.utils.interop import cast_int_addr
        address = cast_int_addr(address)
        return cls(shadow=address)

    def close(self, linger=None):
        """
        Close the socket.

        If linger is specified, LINGER sockopt will be set prior to closing.

        Note: closing a zmq Socket may not close the underlying sockets
        if there are undelivered messages.
        Only after all messages are delivered or discarded by reaching the socket's LINGER timeout
        (default: forever)
        will the underlying sockets be closed.

        This can be called to close the socket by hand. If this is not
        called, the socket will automatically be closed when it is
        garbage collected.
        """
        if self.context:
            self.context._rm_socket(self)
        super(Socket, self).close(linger=linger)

    #-------------------------------------------------------------------------
    # Deprecated aliases
    #-------------------------------------------------------------------------

    @property
    def socket_type(self):
        warnings.warn("Socket.socket_type is deprecated, use Socket.type",
            DeprecationWarning
        )
        return self.type
    
    #-------------------------------------------------------------------------
    # Hooks for sockopt completion
    #-------------------------------------------------------------------------

    def __dir__(self):
        keys = dir(self.__class__)
        for collection in (
            bytes_sockopt_names,
            int_sockopt_names,
            int64_sockopt_names,
            fd_sockopt_names,
        ):
            keys.extend(collection)
        return keys

    #-------------------------------------------------------------------------
    # Getting/Setting options
    #-------------------------------------------------------------------------
    setsockopt = SocketBase.set
    getsockopt = SocketBase.get

    def __setattr__(self, key, value):
        """Override to allow setting zmq.[UN]SUBSCRIBE even though we have a subscribe method"""
        if key in self.__dict__:
            object.__setattr__(self, key, value)
            return
        _key = key.lower()
        if _key in ('subscribe', 'unsubscribe'):

            if isinstance(value, unicode):
                value = value.encode('utf8')
            if _key == 'subscribe':
                self.set(zmq.SUBSCRIBE, value)
            else:
                self.set(zmq.UNSUBSCRIBE, value)
            return
        super(Socket, self).__setattr__(key, value)

    def fileno(self):
        """Return edge-triggered file descriptor for this socket.

        This is a read-only edge-triggered file descriptor for both read and write events on this socket.
        It is important that all available events be consumed when an event is detected,
        otherwise the read event will not trigger again.

        .. versionadded:: 17.0
        """
        return self.FD

    def subscribe(self, topic):
        """Subscribe to a topic

        Only for SUB sockets.

        .. versionadded:: 15.3
        """
        if isinstance(topic, unicode):
            topic = topic.encode('utf8')
        self.set(zmq.SUBSCRIBE, topic)
    
    def unsubscribe(self, topic):
        """Unsubscribe from a topic

        Only for SUB sockets.

        .. versionadded:: 15.3
        """
        if isinstance(topic, unicode):
            topic = topic.encode('utf8')
        self.set(zmq.UNSUBSCRIBE, topic)
    
    def set_string(self, option, optval, encoding='utf-8'):
        """Set socket options with a unicode object.
        
        This is simply a wrapper for setsockopt to protect from encoding ambiguity.

        See the 0MQ documentation for details on specific options.
        
        Parameters
        ----------
        option : int
            The name of the option to set. Can be any of: SUBSCRIBE, 
            UNSUBSCRIBE, IDENTITY
        optval : unicode string (unicode on py2, str on py3)
            The value of the option to set.
        encoding : str
            The encoding to be used, default is utf8
        """
        if not isinstance(optval, unicode):
            raise TypeError("unicode strings only")
        return self.set(option, optval.encode(encoding))
    
    setsockopt_unicode = setsockopt_string = set_string
    
    def get_string(self, option, encoding='utf-8'):
        """Get the value of a socket option.

        See the 0MQ documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to retrieve.

        Returns
        -------
        optval : unicode string (unicode on py2, str on py3)
            The value of the option as a unicode string.
        """
    
        if option not in constants.bytes_sockopts:
            raise TypeError("option %i will not return a string to be decoded"%option)
        return self.getsockopt(option).decode(encoding)
    
    getsockopt_unicode = getsockopt_string = get_string
    
    def bind_to_random_port(self, addr, min_port=49152, max_port=65536, max_tries=100):
        """Bind this socket to a random port in a range.

        If the port range is unspecified, the system will choose the port.

        Parameters
        ----------
        addr : str
            The address string without the port to pass to ``Socket.bind()``.
        min_port : int, optional
            The minimum port in the range of ports to try (inclusive).
        max_port : int, optional
            The maximum port in the range of ports to try (exclusive).
        max_tries : int, optional
            The maximum number of bind attempts to make.

        Returns
        -------
        port : int
            The port the socket was bound to.
    
        Raises
        ------
        ZMQBindError
            if `max_tries` reached before successful bind
        """
        if hasattr(constants, 'LAST_ENDPOINT') and min_port == 49152 and max_port == 65536:
            # if LAST_ENDPOINT is supported, and min_port / max_port weren't specified,
            # we can bind to port 0 and let the OS do the work
            self.bind("%s:*" % addr)
            url = self.last_endpoint.decode('ascii', 'replace')
            _, port_s = url.rsplit(':', 1)
            return int(port_s)
        
        for i in range(max_tries):
            try:
                port = random.randrange(min_port, max_port)
                self.bind('%s:%s' % (addr, port))
            except ZMQError as exception:
                en = exception.errno
                if en == zmq.EADDRINUSE:
                    continue
                elif sys.platform == 'win32' and en == errno.EACCES:
                    continue
                else:
                    raise
            else:
                return port
        raise ZMQBindError("Could not bind socket to random port.")
    
    def get_hwm(self):
        """Get the High Water Mark.
        
        On libzmq ≥ 3, this gets SNDHWM if available, otherwise RCVHWM
        """
        major = zmq.zmq_version_info()[0]
        if major >= 3:
            # return sndhwm, fallback on rcvhwm
            try:
                return self.getsockopt(zmq.SNDHWM)
            except zmq.ZMQError:
                pass
            
            return self.getsockopt(zmq.RCVHWM)
        else:
            return self.getsockopt(zmq.HWM)
    
    def set_hwm(self, value):
        """Set the High Water Mark.
        
        On libzmq ≥ 3, this sets both SNDHWM and RCVHWM


        .. warning::

            New values only take effect for subsequent socket
            bind/connects.
        """
        major = zmq.zmq_version_info()[0]
        if major >= 3:
            raised = None
            try:
                self.sndhwm = value
            except Exception as e:
                raised = e
            try:
                self.rcvhwm = value
            except Exception as e:
                raised = e
            
            if raised:
                raise raised
        else:
            return self.setsockopt(zmq.HWM, value)
    
    hwm = property(get_hwm, set_hwm,
        """Property for High Water Mark.
        
        Setting hwm sets both SNDHWM and RCVHWM as appropriate.
        It gets SNDHWM if available, otherwise RCVHWM.
        """
    )
    
    #-------------------------------------------------------------------------
    # Sending and receiving messages
    #-------------------------------------------------------------------------
    
    def send(self, data, flags=0, copy=True, track=False, routing_id=None, group=None):
        """Send a single zmq message frame on this socket.

        This queues the message to be sent by the IO thread at a later time.

        With flags=NOBLOCK, this raises :class:`ZMQError` if the queue is full;
        otherwise, this waits until space is available.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        data : bytes, Frame, memoryview
            The content of the message. This can be any object that provides
            the Python buffer API (i.e. `memoryview(data)` can be called).
        flags : int
            0, NOBLOCK, SNDMORE, or NOBLOCK|SNDMORE.
        copy : bool
            Should the message be sent in a copying or non-copying manner.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)
        routing_id : int
            For use with SERVER sockets
        group : str
            For use with RADIO sockets

        Returns
        -------
        None : if `copy` or not track
            None if message was sent, raises an exception otherwise.
        MessageTracker : if track and not copy
            a MessageTracker object, whose `pending` property will
            be True until the send is completed.

        Raises
        ------
        TypeError
            If a unicode object is passed
        ValueError
            If `track=True`, but an untracked Frame is passed.
        ZMQError
            If the send does not succeed for any reason (including
            if NOBLOCK is set and the outgoing queue is full).


        .. versionchanged:: 17.0

            DRAFT support for routing_id and group arguments.
        """
        if routing_id is not None:
            if not isinstance(data, zmq.Frame):
                data = zmq.Frame(data, track=track, copy=copy or None,
                                 copy_threshold=self.copy_threshold)
            data.routing_id = routing_id
        if group is not None:
            if not isinstance(data, zmq.Frame):
                data = zmq.Frame(data, track=track, copy=copy or None,
                                 copy_threshold=self.copy_threshold)
            data.group = group
        return super(Socket, self).send(data, flags=flags, copy=copy, track=track)

    def send_multipart(self, msg_parts, flags=0, copy=True, track=False, **kwargs):
        """Send a sequence of buffers as a multipart message.

        The zmq.SNDMORE flag is added to all msg parts before the last.

        Parameters
        ----------
        msg_parts : iterable
            A sequence of objects to send as a multipart message. Each element
            can be any sendable object (Frame, bytes, buffer-providers)
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
            SNDMORE is added automatically for frames before the last.
        copy : bool, optional
            Should the frame(s) be sent in a copying or non-copying manner.
            If copy=False, frames smaller than self.copy_threshold bytes
            will be copied anyway.
        track : bool, optional
            Should the frame(s) be tracked for notification that ZMQ has
            finished with it (ignored if copy=True).
    
        Returns
        -------
        None : if copy or not track
        MessageTracker : if track and not copy
            a MessageTracker object, whose `pending` property will
            be True until the last send is completed.
        """
        # typecheck parts before sending:
        for i,msg in enumerate(msg_parts):
            if isinstance(msg, (zmq.Frame, bytes, memoryview)):
                continue
            try:
                memoryview(msg)
            except Exception:
                rmsg = repr(msg)
                if len(rmsg) > 32:
                    rmsg = rmsg[:32] + '...'
                raise TypeError(
                    "Frame %i (%s) does not support the buffer interface." % (
                    i, rmsg,
                ))
        for msg in msg_parts[:-1]:
            self.send(msg, SNDMORE|flags, copy=copy, track=track)
        # Send the last part without the extra SNDMORE flag.
        return self.send(msg_parts[-1], flags, copy=copy, track=track)

    def recv_multipart(self, flags=0, copy=True, track=False):
        """Receive a multipart message as a list of bytes or Frame objects

        Parameters
        ----------
        flags : int, optional
            Any valid flags for :func:`Socket.recv`.
        copy : bool, optional
            Should the message frame(s) be received in a copying or non-copying manner?
            If False a Frame object is returned for each part, if True a copy of
            the bytes is made for each frame.
        track : bool, optional
            Should the message frame(s) be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)
        
        Returns
        -------
        msg_parts : list
            A list of frames in the multipart message; either Frames or bytes,
            depending on `copy`.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        parts = [self.recv(flags, copy=copy, track=track)]
        # have first part already, only loop while more to receive
        while self.getsockopt(zmq.RCVMORE):
            part = self.recv(flags, copy=copy, track=track)
            parts.append(part)
    
        return parts

    def _deserialize(self, recvd, load):
        """Deserialize a received message

        Override in subclass (e.g. Futures) if recvd is not the raw bytes.

        The default implementation expects bytes and returns the deserialized message immediately.

        Parameters
        ----------

        load: callable
            Callable that deserializes bytes
        recvd:
            The object returned by self.recv

        """
        return load(recvd)

    def send_serialized(self, msg, serialize, flags=0, copy=True, **kwargs):
        """Send a message with a custom serialization function.

        .. versionadded:: 17

        Parameters
        ----------
        msg : The message to be sent. Can be any object serializable by `serialize`.
        serialize : callable
            The serialization function to use.
            serialize(msg) should return an iterable of sendable message frames
            (e.g. bytes objects), which will be passed to send_multipart.
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
        copy : bool, optional
            Whether to copy the frames.

        """
        frames = serialize(msg)
        return self.send_multipart(frames, flags=flags, copy=copy, **kwargs)

    def recv_serialized(self, deserialize, flags=0, copy=True):
        """Receive a message with a custom deserialization function.

        .. versionadded:: 17

        Parameters
        ----------
        deserialize : callable
            The deserialization function to use.
            deserialize will be called with one argument: the list of frames
            returned by recv_multipart() and can return any object.
        flags : int, optional
            Any valid flags for :func:`Socket.recv`.
        copy : bool, optional
            Whether to recv bytes or Frame objects.

        Returns
        -------
        obj : object
            The object returned by the deserialization function.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        frames = self.recv_multipart(flags=flags, copy=copy)
        return self._deserialize(frames, deserialize)

    def send_string(self, u, flags=0, copy=True, encoding='utf-8', **kwargs):
        """Send a Python unicode string as a message with an encoding.
    
        0MQ communicates with raw bytes, so you must encode/decode
        text (unicode on py2, str on py3) around 0MQ.
        
        Parameters
        ----------
        u : Python unicode string (unicode on py2, str on py3)
            The unicode string to send.
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
        encoding : str [default: 'utf-8']
            The encoding to be used
        """
        if not isinstance(u, basestring):
            raise TypeError("unicode/str objects only")
        return self.send(u.encode(encoding), flags=flags, copy=copy, **kwargs)
    
    send_unicode = send_string
    
    def recv_string(self, flags=0, encoding='utf-8'):
        """Receive a unicode string, as sent by send_string.
    
        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.
        encoding : str [default: 'utf-8']
            The encoding to be used

        Returns
        -------
        s : unicode string (unicode on py2, str on py3)
            The Python unicode string that arrives as encoded bytes.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags=flags)
        return self._deserialize(msg, lambda buf: buf.decode(encoding))
    
    recv_unicode = recv_string
    
    def send_pyobj(self, obj, flags=0, protocol=DEFAULT_PROTOCOL, **kwargs):
        """Send a Python object as a message using pickle to serialize.

        Parameters
        ----------
        obj : Python object
            The Python object to send.
        flags : int
            Any valid flags for :func:`Socket.send`.
        protocol : int
            The pickle protocol number to use. The default is pickle.DEFAULT_PROTOCOL
            where defined, and pickle.HIGHEST_PROTOCOL elsewhere.
        """
        msg = pickle.dumps(obj, protocol)
        return self.send(msg, flags=flags, **kwargs)

    def recv_pyobj(self, flags=0):
        """Receive a Python object as a message using pickle to serialize.

        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.

        Returns
        -------
        obj : Python object
            The Python object that arrives as a message.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags)
        return self._deserialize(msg, pickle.loads)

    def send_json(self, obj, flags=0, **kwargs):
        """Send a Python object as a message using json to serialize.
        
        Keyword arguments are passed on to json.dumps
        
        Parameters
        ----------
        obj : Python object
            The Python object to send
        flags : int
            Any valid flags for :func:`Socket.send`
        """
        send_kwargs = {}
        for key in ('routing_id', 'group'):
            if key in kwargs:
                send_kwargs[key] = kwargs.pop(key)
        msg = jsonapi.dumps(obj, **kwargs)
        return self.send(msg, flags=flags, **send_kwargs)

    def recv_json(self, flags=0, **kwargs):
        """Receive a Python object as a message using json to serialize.

        Keyword arguments are passed on to json.loads
        
        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.

        Returns
        -------
        obj : Python object
            The Python object that arrives as a message.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags)
        return self._deserialize(msg, lambda buf: jsonapi.loads(buf, **kwargs))

    _poller_class = Poller

    def poll(self, timeout=None, flags=POLLIN):
        """Poll the socket for events.
        See :class:`Poller` to wait for multiple sockets at once.

        Parameters
        ----------
        timeout : int [default: None]
            The timeout (in milliseconds) to wait for an event. If unspecified
            (or specified None), will wait forever for an event.
        flags : int [default: POLLIN]
            POLLIN, POLLOUT, or POLLIN|POLLOUT. The event flags to poll for.

        Returns
        -------
        event_mask : int
            The poll event mask (POLLIN, POLLOUT),
            0 if the timeout was reached without an event.
        """

        if self.closed:
            raise ZMQError(ENOTSUP)

        p = self._poller_class()
        p.register(self, flags)
        evts = dict(p.poll(timeout))
        # return 0 if no events, otherwise return event bitfield
        return evts.get(self, 0)

    def get_monitor_socket(self, events=None, addr=None):
        """Return a connected PAIR socket ready to receive the event notifications.

        .. versionadded:: libzmq-4.0
        .. versionadded:: 14.0

        Parameters
        ----------
        events : int [default: ZMQ_EVENT_ALL]
            The bitmask defining which events are wanted.
        addr :  string [default: None]
            The optional endpoint for the monitoring sockets.

        Returns
        -------
        socket :  (PAIR)
            The socket is already connected and ready to receive messages.
        """
        # safe-guard, method only available on libzmq >= 4
        if zmq.zmq_version_info() < (4,):
            raise NotImplementedError("get_monitor_socket requires libzmq >= 4, have %s" % zmq.zmq_version())

        # if already monitoring, return existing socket
        if self._monitor_socket:
            if self._monitor_socket.closed:
                self._monitor_socket = None
            else:
                return self._monitor_socket

        if addr is None:
            # create endpoint name from internal fd
            addr = "inproc://monitor.s-%d" % self.FD
        if events is None:
            # use all events
            events = zmq.EVENT_ALL
        # attach monitoring socket
        self.monitor(addr, events)
        # create new PAIR socket and connect it
        self._monitor_socket = self.context.socket(zmq.PAIR)
        self._monitor_socket.connect(addr)
        return self._monitor_socket

    def disable_monitor(self):
        """Shutdown the PAIR socket (created using get_monitor_socket)
        that is serving socket events.
        
        .. versionadded:: 14.4
        """
        self._monitor_socket = None
        self.monitor(None, 0)


__all__ = ['Socket']
