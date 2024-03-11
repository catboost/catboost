"""Base class to manage a running kernel"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import absolute_import

from contextlib import contextmanager
import os
import re
import signal
import sys
import time
import warnings

import zmq

from ipython_genutils.importstring import import_item
from .localinterfaces import is_local_ip, local_ips
from traitlets import (
    Any, Float, Instance, Unicode, List, Bool, Type, DottedObjectName
)
from jupyter_client import (
    launch_kernel,
    kernelspec,
)
from .connect import ConnectionFileMixin
from .managerabc import (
    KernelManagerABC
)


class KernelManager(ConnectionFileMixin):
    """Manages a single kernel in a subprocess on this host.

    This version starts kernels with Popen.
    """

    # The PyZMQ Context to use for communication with the kernel.
    context = Instance(zmq.Context)
    def _context_default(self):
        return zmq.Context()

    # the class to create with our `client` method
    client_class = DottedObjectName('jupyter_client.blocking.BlockingKernelClient')
    client_factory = Type(klass='jupyter_client.KernelClient')
    def _client_factory_default(self):
        return import_item(self.client_class)

    def _client_class_changed(self, name, old, new):
        self.client_factory = import_item(str(new))

    # The kernel process with which the KernelManager is communicating.
    # generally a Popen instance
    kernel = Any()

    kernel_spec_manager = Instance(kernelspec.KernelSpecManager)

    def _kernel_spec_manager_default(self):
        return kernelspec.KernelSpecManager(data_dir=self.data_dir)

    def _kernel_spec_manager_changed(self):
        self._kernel_spec = None

    shutdown_wait_time = Float(
        5.0, config=True,
        help="Time to wait for a kernel to terminate before killing it, "
             "in seconds.")

    kernel_name = Unicode(kernelspec.NATIVE_KERNEL_NAME)

    def _kernel_name_changed(self, name, old, new):
        self._kernel_spec = None
        if new == 'python':
            self.kernel_name = kernelspec.NATIVE_KERNEL_NAME

    _kernel_spec = None

    @property
    def kernel_spec(self):
        if self._kernel_spec is None and self.kernel_name is not '':
            self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
        return self._kernel_spec

    kernel_cmd = List(Unicode(), config=True,
        help="""DEPRECATED: Use kernel_name instead.

        The Popen Command to launch the kernel.
        Override this if you have a custom kernel.
        If kernel_cmd is specified in a configuration file,
        Jupyter does not pass any arguments to the kernel,
        because it cannot make any assumptions about the
        arguments that the kernel understands. In particular,
        this means that the kernel does not receive the
        option --debug if it given on the Jupyter command line.
        """
    )

    def _kernel_cmd_changed(self, name, old, new):
        warnings.warn("Setting kernel_cmd is deprecated, use kernel_spec to "
                      "start different kernels.")

    @property
    def ipykernel(self):
        return self.kernel_name in {'python', 'python2', 'python3'}

    # Protected traits
    _launch_args = Any()
    _control_socket = Any()

    _restarter = Any()

    autorestart = Bool(True, config=True,
        help="""Should we autorestart the kernel if it dies."""
    )

    def __del__(self):
        self._close_control_socket()
        self.cleanup_connection_file()

    #--------------------------------------------------------------------------
    # Kernel restarter
    #--------------------------------------------------------------------------

    def start_restarter(self):
        pass

    def stop_restarter(self):
        pass

    def add_restart_callback(self, callback, event='restart'):
        """register a callback to be called when a kernel is restarted"""
        if self._restarter is None:
            return
        self._restarter.add_callback(callback, event)

    def remove_restart_callback(self, callback, event='restart'):
        """unregister a callback to be called when a kernel is restarted"""
        if self._restarter is None:
            return
        self._restarter.remove_callback(callback, event)

    #--------------------------------------------------------------------------
    # create a Client connected to our Kernel
    #--------------------------------------------------------------------------

    def client(self, **kwargs):
        """Create a client configured to connect to our kernel"""
        kw = {}
        kw.update(self.get_connection_info(session=True))
        kw.update(dict(
            connection_file=self.connection_file,
            parent=self,
        ))

        # add kwargs last, for manual overrides
        kw.update(kwargs)
        return self.client_factory(**kw)

    #--------------------------------------------------------------------------
    # Kernel management
    #--------------------------------------------------------------------------

    def format_kernel_cmd(self, extra_arguments=None):
        """replace templated args (e.g. {connection_file})"""
        extra_arguments = extra_arguments or []
        if self.kernel_cmd:
            cmd = self.kernel_cmd + extra_arguments
        else:
            cmd = self.kernel_spec.argv + extra_arguments

        if cmd and cmd[0] in {'python',
                              'python%i' % sys.version_info[0],
                              'python%i.%i' % sys.version_info[:2]}:
            # executable is 'python' or 'python3', use sys.executable.
            # These will typically be the same,
            # but if the current process is in an env
            # and has been launched by abspath without
            # activating the env, python on PATH may not be sys.executable,
            # but it should be.
            cmd[0] = sys.executable

        ns = dict(connection_file=self.connection_file,
                  prefix=sys.prefix,
                 )

        if self.kernel_spec:
            ns["resource_dir"] = self.kernel_spec.resource_dir

        ns.update(self._launch_args)

        pat = re.compile(r'\{([A-Za-z0-9_]+)\}')
        def from_ns(match):
            """Get the key out of ns if it's there, otherwise no change."""
            return ns.get(match.group(1), match.group())

        return [ pat.sub(from_ns, arg) for arg in cmd ]

    def _launch_kernel(self, kernel_cmd, **kw):
        """actually launch the kernel

        override in a subclass to launch kernel subprocesses differently
        """
        return launch_kernel(kernel_cmd, **kw)

    # Control socket used for polite kernel shutdown

    def _connect_control_socket(self):
        if self._control_socket is None:
            self._control_socket = self._create_connected_socket('control')
            self._control_socket.linger = 100

    def _close_control_socket(self):
        if self._control_socket is None:
            return
        self._control_socket.close()
        self._control_socket = None

    def start_kernel(self, **kw):
        """Starts a kernel on this host in a separate process.

        If random ports (port=0) are being used, this method must be called
        before the channels are created.

        Parameters
        ----------
        `**kw` : optional
             keyword arguments that are passed down to build the kernel_cmd
             and launching the kernel (e.g. Popen kwargs).
        """
        if self.transport == 'tcp' and not is_local_ip(self.ip):
            raise RuntimeError("Can only launch a kernel on a local interface. "
                               "This one is not: %s."
                               "Make sure that the '*_address' attributes are "
                               "configured properly. "
                               "Currently valid addresses are: %s" % (self.ip, local_ips())
                               )

        # write connection file / get default ports
        self.write_connection_file()

        # save kwargs for use in restart
        self._launch_args = kw.copy()
        # build the Popen cmd
        extra_arguments = kw.pop('extra_arguments', [])
        kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
        env = kw.pop('env', os.environ).copy()
        # Don't allow PYTHONEXECUTABLE to be passed to kernel process.
        # If set, it can bork all the things.
        env.pop('PYTHONEXECUTABLE', None)
        if not self.kernel_cmd:
            # If kernel_cmd has been set manually, don't refer to a kernel spec
            # Environment variables from kernel spec are added to os.environ
            env.update(self.kernel_spec.env or {})

        # launch the kernel subprocess
        self.log.debug("Starting kernel: %s", kernel_cmd)
        self.kernel = self._launch_kernel(kernel_cmd, env=env,
                                    **kw)
        self.start_restarter()
        self._connect_control_socket()

    def request_shutdown(self, restart=False):
        """Send a shutdown request via control channel
        """
        content = dict(restart=restart)
        msg = self.session.msg("shutdown_request", content=content)
        # ensure control socket is connected
        self._connect_control_socket()
        self.session.send(self._control_socket, msg)

    def finish_shutdown(self, waittime=None, pollinterval=0.1):
        """Wait for kernel shutdown, then kill process if it doesn't shutdown.

        This does not send shutdown requests - use :meth:`request_shutdown`
        first.
        """
        if waittime is None:
            waittime = max(self.shutdown_wait_time, 0)
        for i in range(int(waittime/pollinterval)):
            if self.is_alive():
                time.sleep(pollinterval)
            else:
                break
        else:
            # OK, we've waited long enough.
            if self.has_kernel:
                self.log.debug("Kernel is taking too long to finish, killing")
                self._kill_kernel()

    def cleanup(self, connection_file=True):
        """Clean up resources when the kernel is shut down"""
        if connection_file:
            self.cleanup_connection_file()

        self.cleanup_ipc_files()
        self._close_control_socket()

    def shutdown_kernel(self, now=False, restart=False):
        """Attempts to stop the kernel process cleanly.

        This attempts to shutdown the kernels cleanly by:

        1. Sending it a shutdown message over the shell channel.
        2. If that fails, the kernel is shutdown forcibly by sending it
           a signal.

        Parameters
        ----------
        now : bool
            Should the kernel be forcible killed *now*. This skips the
            first, nice shutdown attempt.
        restart: bool
            Will this kernel be restarted after it is shutdown. When this
            is True, connection files will not be cleaned up.
        """
        # Stop monitoring for restarting while we shutdown.
        self.stop_restarter()

        if now:
            self._kill_kernel()
        else:
            self.request_shutdown(restart=restart)
            # Don't send any additional kernel kill messages immediately, to give
            # the kernel a chance to properly execute shutdown actions. Wait for at
            # most 1s, checking every 0.1s.
            self.finish_shutdown()

        self.cleanup(connection_file=not restart)

    def restart_kernel(self, now=False, newports=False, **kw):
        """Restarts a kernel with the arguments that were used to launch it.

        Parameters
        ----------
        now : bool, optional
            If True, the kernel is forcefully restarted *immediately*, without
            having a chance to do any cleanup action.  Otherwise the kernel is
            given 1s to clean up before a forceful restart is issued.

            In all cases the kernel is restarted, the only difference is whether
            it is given a chance to perform a clean shutdown or not.

        newports : bool, optional
            If the old kernel was launched with random ports, this flag decides
            whether the same ports and connection file will be used again.
            If False, the same ports and connection file are used. This is
            the default. If True, new random port numbers are chosen and a
            new connection file is written. It is still possible that the newly
            chosen random port numbers happen to be the same as the old ones.

        `**kw` : optional
            Any options specified here will overwrite those used to launch the
            kernel.
        """
        if self._launch_args is None:
            raise RuntimeError("Cannot restart the kernel. "
                               "No previous call to 'start_kernel'.")
        else:
            # Stop currently running kernel.
            self.shutdown_kernel(now=now, restart=True)

            if newports:
                self.cleanup_random_ports()

            # Start new kernel.
            self._launch_args.update(kw)
            self.start_kernel(**self._launch_args)

    @property
    def has_kernel(self):
        """Has a kernel been started that we are managing."""
        return self.kernel is not None

    def _kill_kernel(self):
        """Kill the running kernel.

        This is a private method, callers should use shutdown_kernel(now=True).
        """
        if self.has_kernel:

            # Signal the kernel to terminate (sends SIGKILL on Unix and calls
            # TerminateProcess() on Win32).
            try:
                if hasattr(signal, 'SIGKILL'):
                    self.signal_kernel(signal.SIGKILL)
                else:
                    self.kernel.kill()
            except OSError as e:
                # In Windows, we will get an Access Denied error if the process
                # has already terminated. Ignore it.
                if sys.platform == 'win32':
                    if e.winerror != 5:
                        raise
                # On Unix, we may get an ESRCH error if the process has already
                # terminated. Ignore it.
                else:
                    from errno import ESRCH
                    if e.errno != ESRCH:
                        raise

            # Block until the kernel terminates.
            self.kernel.wait()
            self.kernel = None
        else:
            raise RuntimeError("Cannot kill kernel. No kernel is running!")

    def interrupt_kernel(self):
        """Interrupts the kernel by sending it a signal.

        Unlike ``signal_kernel``, this operation is well supported on all
        platforms.
        """
        if self.has_kernel:
            interrupt_mode = self.kernel_spec.interrupt_mode
            if interrupt_mode == 'signal':
                if sys.platform == 'win32':
                    from .win_interrupt import send_interrupt
                    send_interrupt(self.kernel.win32_interrupt_event)
                else:
                    self.signal_kernel(signal.SIGINT)

            elif interrupt_mode == 'message':
                msg = self.session.msg("interrupt_request", content={})
                self._connect_control_socket()
                self.session.send(self._control_socket, msg)
        else:
            raise RuntimeError("Cannot interrupt kernel. No kernel is running!")

    def signal_kernel(self, signum):
        """Sends a signal to the process group of the kernel (this
        usually includes the kernel and any subprocesses spawned by
        the kernel).

        Note that since only SIGTERM is supported on Windows, this function is
        only useful on Unix systems.
        """
        if self.has_kernel:
            if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(self.kernel.pid)
                    os.killpg(pgid, signum)
                    return
                except OSError:
                    pass
            self.kernel.send_signal(signum)
        else:
            raise RuntimeError("Cannot signal kernel. No kernel is running!")

    def is_alive(self):
        """Is the kernel process still running?"""
        if self.has_kernel:
            if self.kernel.poll() is None:
                return True
            else:
                return False
        else:
            # we don't have a kernel
            return False


KernelManagerABC.register(KernelManager)


def start_new_kernel(startup_timeout=60, kernel_name='python', **kwargs):
    """Start a new kernel, and return its Manager and Client"""
    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel(**kwargs)
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=startup_timeout)
    except RuntimeError:
        kc.stop_channels()
        km.shutdown_kernel()
        raise

    return km, kc

@contextmanager
def run_kernel(**kwargs):
    """Context manager to create a kernel in a subprocess.

    The kernel is shut down when the context exits.

    Returns
    -------
    kernel_client: connected KernelClient instance
    """
    km, kc = start_new_kernel(**kwargs)
    try:
        yield kc
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
