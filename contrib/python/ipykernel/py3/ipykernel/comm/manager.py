"""Base class to manage comms"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import logging

from traitlets import Dict, Instance
from traitlets.config import LoggingConfigurable
from traitlets.utils.importstring import import_item

from .comm import Comm


class CommManager(LoggingConfigurable):
    """Manager for Comms in the Kernel"""

    kernel = Instance("ipykernel.kernelbase.Kernel")
    comms = Dict()
    targets = Dict()

    # Public APIs

    def register_target(self, target_name, f):
        """Register a callable f for a given target name

        f will be called with two arguments when a comm_open message is received with `target`:

        - the Comm instance
        - the `comm_open` message itself.

        f can be a Python callable or an import string for one.
        """
        if isinstance(f, str):
            f = import_item(f)

        self.targets[target_name] = f

    def unregister_target(self, target_name, f):
        """Unregister a callable registered with register_target"""
        return self.targets.pop(target_name)

    def register_comm(self, comm):
        """Register a new comm"""
        comm_id = comm.comm_id
        comm.kernel = self.kernel
        self.comms[comm_id] = comm
        return comm_id

    def unregister_comm(self, comm):
        """Unregister a comm, and close its counterpart"""
        # unlike get_comm, this should raise a KeyError
        comm = self.comms.pop(comm.comm_id)

    def get_comm(self, comm_id):
        """Get a comm with a particular id

        Returns the comm if found, otherwise None.

        This will not raise an error,
        it will log messages if the comm cannot be found.
        """
        try:
            return self.comms[comm_id]
        except KeyError:
            self.log.warning("No such comm: %s", comm_id)
            if self.log.isEnabledFor(logging.DEBUG):
                # don't create the list of keys if debug messages aren't enabled
                self.log.debug("Current comms: %s", list(self.comms.keys()))

    # Message handlers
    def comm_open(self, stream, ident, msg):
        """Handler for comm_open messages"""
        content = msg["content"]
        comm_id = content["comm_id"]
        target_name = content["target_name"]
        f = self.targets.get(target_name, None)
        comm = Comm(
            comm_id=comm_id,
            primary=False,
            target_name=target_name,
        )
        self.register_comm(comm)
        if f is None:
            self.log.error("No such comm target registered: %s", target_name)
        else:
            try:
                f(comm, msg)
                return
            except Exception:
                self.log.error("Exception opening comm with target: %s", target_name, exc_info=True)

        # Failure.
        try:
            comm.close()
        except Exception:
            self.log.error(
                """Could not close comm during `comm_open` failure
                clean-up.  The comm may not have been opened yet.""",
                exc_info=True,
            )

    def comm_msg(self, stream, ident, msg):
        """Handler for comm_msg messages"""
        content = msg["content"]
        comm_id = content["comm_id"]
        comm = self.get_comm(comm_id)
        if comm is None:
            return

        try:
            comm.handle_msg(msg)
        except Exception:
            self.log.error("Exception in comm_msg for %s", comm_id, exc_info=True)

    def comm_close(self, stream, ident, msg):
        """Handler for comm_close messages"""
        content = msg["content"]
        comm_id = content["comm_id"]
        comm = self.get_comm(comm_id)
        if comm is None:
            return

        self.comms[comm_id]._closed = True
        del self.comms[comm_id]

        try:
            comm.handle_close(msg)
        except Exception:
            self.log.error("Exception in comm_close for %s", comm_id, exc_info=True)


__all__ = ["CommManager"]
