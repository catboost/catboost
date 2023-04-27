"""Base class for a Comm"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import uuid

from traitlets.config import LoggingConfigurable
from ipykernel.kernelbase import Kernel

from ipykernel.jsonutil import json_clean
from traitlets import Instance, Unicode, Bytes, Bool, Dict, Any, default


class Comm(LoggingConfigurable):
    """Class for communicating between a Frontend and a Kernel"""
    kernel = Instance('ipykernel.kernelbase.Kernel', allow_none=True)

    @default('kernel')
    def _default_kernel(self):
        if Kernel.initialized():
            return Kernel.instance()

    comm_id = Unicode()

    @default('comm_id')
    def _default_comm_id(self):
        return uuid.uuid4().hex

    primary = Bool(True, help="Am I the primary or secondary Comm?")

    target_name = Unicode('comm')
    target_module = Unicode(None, allow_none=True, help="""requirejs module from
        which to load comm target.""")

    topic = Bytes()

    @default('topic')
    def _default_topic(self):
        return ('comm-%s' % self.comm_id).encode('ascii')

    _open_data = Dict(help="data dict, if any, to be included in comm_open")
    _close_data = Dict(help="data dict, if any, to be included in comm_close")

    _msg_callback = Any()
    _close_callback = Any()

    _closed = Bool(True)

    def __init__(self, target_name='', data=None, metadata=None, buffers=None, **kwargs):
        if target_name:
            kwargs['target_name'] = target_name
        super(Comm, self).__init__(**kwargs)
        if self.kernel:
            if self.primary:
                # I am primary, open my peer.
                self.open(data=data, metadata=metadata, buffers=buffers)
            else:
                self._closed = False

    def _publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **keys):
        """Helper for sending a comm message on IOPub"""
        data = {} if data is None else data
        metadata = {} if metadata is None else metadata
        content = json_clean(dict(data=data, comm_id=self.comm_id, **keys))
        self.kernel.session.send(self.kernel.iopub_socket, msg_type,
            content,
            metadata=json_clean(metadata),
            parent=self.kernel._parent_header,
            ident=self.topic,
            buffers=buffers,
        )

    def __del__(self):
        """trigger close on gc"""
        self.close()

    # publishing messages

    def open(self, data=None, metadata=None, buffers=None):
        """Open the frontend-side version of this comm"""
        if data is None:
            data = self._open_data
        comm_manager = getattr(self.kernel, 'comm_manager', None)
        if comm_manager is None:
            raise RuntimeError("Comms cannot be opened without a kernel "
                        "and a comm_manager attached to that kernel.")

        comm_manager.register_comm(self)
        try:
            self._publish_msg('comm_open',
                              data=data, metadata=metadata, buffers=buffers,
                              target_name=self.target_name,
                              target_module=self.target_module,
                              )
            self._closed = False
        except:
            comm_manager.unregister_comm(self)
            raise

    def close(self, data=None, metadata=None, buffers=None):
        """Close the frontend-side version of this comm"""
        if self._closed:
            # only close once
            return
        self._closed = True
        # nothing to send if we have no kernel
        # can be None during interpreter cleanup
        if not self.kernel:
            return
        if data is None:
            data = self._close_data
        self._publish_msg('comm_close',
            data=data, metadata=metadata, buffers=buffers,
        )
        self.kernel.comm_manager.unregister_comm(self)

    def send(self, data=None, metadata=None, buffers=None):
        """Send a message to the frontend-side version of this comm"""
        self._publish_msg('comm_msg',
            data=data, metadata=metadata, buffers=buffers,
        )

    # registering callbacks

    def on_close(self, callback):
        """Register a callback for comm_close

        Will be called with the `data` of the close message.

        Call `on_close(None)` to disable an existing callback.
        """
        self._close_callback = callback

    def on_msg(self, callback):
        """Register a callback for comm_msg

        Will be called with the `data` of any comm_msg messages.

        Call `on_msg(None)` to disable an existing callback.
        """
        self._msg_callback = callback

    # handling of incoming messages

    def handle_close(self, msg):
        """Handle a comm_close message"""
        self.log.debug("handle_close[%s](%s)", self.comm_id, msg)
        if self._close_callback:
            self._close_callback(msg)

    def handle_msg(self, msg):
        """Handle a comm_msg message"""
        self.log.debug("handle_msg[%s](%s)", self.comm_id, msg)
        if self._msg_callback:
            shell = self.kernel.shell
            if shell:
                shell.events.trigger('pre_execute')
            self._msg_callback(msg)
            if shell:
                shell.events.trigger('post_execute')


__all__ = ['Comm']
