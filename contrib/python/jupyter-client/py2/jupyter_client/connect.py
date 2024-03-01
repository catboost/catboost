"""Utilities for connecting to jupyter kernels

The :class:`ConnectionFileMixin` class in this module encapsulates the logic
related to writing and reading connections files.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.


from __future__ import absolute_import

import errno
import glob
import json
import os
import socket
import stat
import tempfile
import warnings
from getpass import getpass
from contextlib import contextmanager

import zmq

from traitlets.config import LoggingConfigurable
from .localinterfaces import localhost
from ipython_genutils.path import filefind
from ipython_genutils.py3compat import (
    bytes_to_str, cast_bytes, cast_bytes_py2, string_types,
)
from traitlets import (
    Bool, Integer, Unicode, CaselessStrEnum, Instance, Type,
)
from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write


def write_connection_file(fname=None, shell_port=0, iopub_port=0, stdin_port=0, hb_port=0,
                         control_port=0, ip='', key=b'', transport='tcp',
                         signature_scheme='hmac-sha256', kernel_name=''
                         ):
    """Generates a JSON config file, including the selection of random ports.

    Parameters
    ----------

    fname : unicode
        The path to the file to write

    shell_port : int, optional
        The port to use for ROUTER (shell) channel.

    iopub_port : int, optional
        The port to use for the SUB channel.

    stdin_port : int, optional
        The port to use for the ROUTER (raw input) channel.

    control_port : int, optional
        The port to use for the ROUTER (control) channel.

    hb_port : int, optional
        The port to use for the heartbeat REP channel.

    ip  : str, optional
        The ip address the kernel will bind to.

    key : str, optional
        The Session key used for message authentication.

    signature_scheme : str, optional
        The scheme used for message authentication.
        This has the form 'digest-hash', where 'digest'
        is the scheme used for digests, and 'hash' is the name of the hash function
        used by the digest scheme.
        Currently, 'hmac' is the only supported digest scheme,
        and 'sha256' is the default hash function.

    kernel_name : str, optional
        The name of the kernel currently connected to.
    """
    if not ip:
        ip = localhost()
    # default to temporary connector file
    if not fname:
        fd, fname = tempfile.mkstemp('.json')
        os.close(fd)

    # Find open ports as necessary.

    ports = []
    ports_needed = int(shell_port <= 0) + \
                   int(iopub_port <= 0) + \
                   int(stdin_port <= 0) + \
                   int(control_port <= 0) + \
                   int(hb_port <= 0)
    if transport == 'tcp':
        for i in range(ports_needed):
            sock = socket.socket()
            # struct.pack('ii', (0,0)) is 8 null bytes
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b'\0' * 8)
            sock.bind((ip, 0))
            ports.append(sock)
        for i, sock in enumerate(ports):
            port = sock.getsockname()[1]
            sock.close()
            ports[i] = port
    else:
        N = 1
        for i in range(ports_needed):
            while os.path.exists("%s-%s" % (ip, str(N))):
                N += 1
            ports.append(N)
            N += 1
    if shell_port <= 0:
        shell_port = ports.pop(0)
    if iopub_port <= 0:
        iopub_port = ports.pop(0)
    if stdin_port <= 0:
        stdin_port = ports.pop(0)
    if control_port <= 0:
        control_port = ports.pop(0)
    if hb_port <= 0:
        hb_port = ports.pop(0)

    cfg = dict( shell_port=shell_port,
                iopub_port=iopub_port,
                stdin_port=stdin_port,
                control_port=control_port,
                hb_port=hb_port,
              )
    cfg['ip'] = ip
    cfg['key'] = bytes_to_str(key)
    cfg['transport'] = transport
    cfg['signature_scheme'] = signature_scheme
    cfg['kernel_name'] = kernel_name

    # Only ever write this file as user read/writeable
    # This would otherwise introduce a vulnerability as a file has secrets
    # which would let others execute arbitrarily code as you
    with secure_write(fname) as f:
        f.write(json.dumps(cfg, indent=2))

    if hasattr(stat, 'S_ISVTX'):
        # set the sticky bit on the file and its parent directory
        # to avoid periodic cleanup
        paths = [fname]
        runtime_dir = os.path.dirname(fname)
        if runtime_dir:
            paths.append(runtime_dir)
        for path in paths:
            permissions = os.stat(path).st_mode
            new_permissions = permissions | stat.S_ISVTX
            if new_permissions != permissions:
                try:
                    os.chmod(path, new_permissions)
                except OSError as e:
                    if e.errno == errno.EPERM and path == runtime_dir:
                        # suppress permission errors setting sticky bit on runtime_dir,
                        # which we may not own.
                        pass
                    else:
                        # failed to set sticky bit, probably not a big deal
                        warnings.warn(
                            "Failed to set sticky bit on %r: %s"
                            "\nProbably not a big deal, but runtime files may be cleaned up periodically." % (path, e),
                            RuntimeWarning,
                        )

    return fname, cfg


def find_connection_file(filename='kernel-*.json', path=None, profile=None):
    """find a connection file, and return its absolute path.

    The current working directory and optional search path
    will be searched for the file if it is not given by absolute path.

    If the argument does not match an existing file, it will be interpreted as a
    fileglob, and the matching file in the profile's security dir with
    the latest access time will be used.

    Parameters
    ----------
    filename : str
        The connection file or fileglob to search for.
    path : str or list of strs[optional]
        Paths in which to search for connection files.

    Returns
    -------
    str : The absolute path of the connection file.
    """
    if profile is not None:
        warnings.warn("Jupyter has no profiles. profile=%s has been ignored." % profile)
    if path is None:
        path = ['.', jupyter_runtime_dir()]
    if isinstance(path, string_types):
        path = [path]

    try:
        # first, try explicit name
        return filefind(filename, path)
    except IOError:
        pass

    # not found by full name

    if '*' in filename:
        # given as a glob already
        pat = filename
    else:
        # accept any substring match
        pat = '*%s*' % filename

    matches = []
    for p in path:
        matches.extend(glob.glob(os.path.join(p, pat)))

    matches = [ os.path.abspath(m) for m in matches ]
    if not matches:
        raise IOError("Could not find %r in %r" % (filename, path))
    elif len(matches) == 1:
        return matches[0]
    else:
        # get most recent match, by access time:
        return sorted(matches, key=lambda f: os.stat(f).st_atime)[-1]


def tunnel_to_kernel(connection_info, sshserver, sshkey=None):
    """tunnel connections to a kernel via ssh

    This will open four SSH tunnels from localhost on this machine to the
    ports associated with the kernel.  They can be either direct
    localhost-localhost tunnels, or if an intermediate server is necessary,
    the kernel must be listening on a public IP.

    Parameters
    ----------
    connection_info : dict or str (path)
        Either a connection dict, or the path to a JSON connection file
    sshserver : str
        The ssh sever to use to tunnel to the kernel. Can be a full
        `user@server:port` string. ssh config aliases are respected.
    sshkey : str [optional]
        Path to file containing ssh key to use for authentication.
        Only necessary if your ssh config does not already associate
        a keyfile with the host.

    Returns
    -------

    (shell, iopub, stdin, hb) : ints
        The four ports on localhost that have been forwarded to the kernel.
    """
    from zmq.ssh import tunnel
    if isinstance(connection_info, string_types):
        # it's a path, unpack it
        with open(connection_info) as f:
            connection_info = json.loads(f.read())

    cf = connection_info

    lports = tunnel.select_random_ports(4)
    rports = cf['shell_port'], cf['iopub_port'], cf['stdin_port'], cf['hb_port']

    remote_ip = cf['ip']

    if tunnel.try_passwordless_ssh(sshserver, sshkey):
        password=False
    else:
        password = getpass("SSH Password for %s: " % cast_bytes_py2(sshserver))

    for lp,rp in zip(lports, rports):
        tunnel.ssh_tunnel(lp, rp, sshserver, remote_ip, sshkey, password)

    return tuple(lports)


#-----------------------------------------------------------------------------
# Mixin for classes that work with connection files
#-----------------------------------------------------------------------------

channel_socket_types = {
    'hb' : zmq.REQ,
    'shell' : zmq.DEALER,
    'iopub' : zmq.SUB,
    'stdin' : zmq.DEALER,
    'control': zmq.DEALER,
}

port_names = [ "%s_port" % channel for channel in ('shell', 'stdin', 'iopub', 'hb', 'control')]

class ConnectionFileMixin(LoggingConfigurable):
    """Mixin for configurable classes that work with connection files"""

    data_dir = Unicode()
    def _data_dir_default(self):
        return jupyter_data_dir()

    # The addresses for the communication channels
    connection_file = Unicode('', config=True,
    help="""JSON file in which to store connection info [default: kernel-<pid>.json]

    This file will contain the IP, ports, and authentication key needed to connect
    clients to this kernel. By default, this file will be created in the security dir
    of the current profile, but can be specified by absolute path.
    """)
    _connection_file_written = Bool(False)

    transport = CaselessStrEnum(['tcp', 'ipc'], default_value='tcp', config=True)
    kernel_name = Unicode()

    ip = Unicode(config=True,
        help="""Set the kernel\'s IP address [default localhost].
        If the IP address is something other than localhost, then
        Consoles on other machines will be able to connect
        to the Kernel, so be careful!"""
    )

    def _ip_default(self):
        if self.transport == 'ipc':
            if self.connection_file:
                return os.path.splitext(self.connection_file)[0] + '-ipc'
            else:
                return 'kernel-ipc'
        else:
            return localhost()

    def _ip_changed(self, name, old, new):
        if new == '*':
            self.ip = '0.0.0.0'

    # protected traits

    hb_port = Integer(0, config=True,
            help="set the heartbeat port [default: random]")
    shell_port = Integer(0, config=True,
            help="set the shell (ROUTER) port [default: random]")
    iopub_port = Integer(0, config=True,
            help="set the iopub (PUB) port [default: random]")
    stdin_port = Integer(0, config=True,
            help="set the stdin (ROUTER) port [default: random]")
    control_port = Integer(0, config=True,
            help="set the control (ROUTER) port [default: random]")

    # names of the ports with random assignment
    _random_port_names = None

    @property
    def ports(self):
        return [ getattr(self, name) for name in port_names ]

    # The Session to use for communication with the kernel.
    session = Instance('jupyter_client.session.Session')
    def _session_default(self):
        from jupyter_client.session import Session
        return Session(parent=self)

    #--------------------------------------------------------------------------
    # Connection and ipc file management
    #--------------------------------------------------------------------------

    def get_connection_info(self, session=False):
        """Return the connection info as a dict

        Parameters
        ----------
        session : bool [default: False]
            If True, return our session object will be included in the connection info.
            If False (default), the configuration parameters of our session object will be included,
            rather than the session object itself.

        Returns
        -------
        connect_info : dict
            dictionary of connection information.
        """
        info = dict(
            transport=self.transport,
            ip=self.ip,
            shell_port=self.shell_port,
            iopub_port=self.iopub_port,
            stdin_port=self.stdin_port,
            hb_port=self.hb_port,
            control_port=self.control_port,
        )
        if session:
            # add *clone* of my session,
            # so that state such as digest_history is not shared.
            info['session'] = self.session.clone()
        else:
            # add session info
            info.update(dict(
                signature_scheme=self.session.signature_scheme,
                key=self.session.key,
            ))
        return info

    # factory for blocking clients
    blocking_class = Type(klass=object, default_value='jupyter_client.BlockingKernelClient')
    def blocking_client(self):
        """Make a blocking client connected to my kernel"""
        info = self.get_connection_info()
        info['parent'] = self
        bc = self.blocking_class(**info)
        bc.session.key = self.session.key
        return bc

    def cleanup_connection_file(self):
        """Cleanup connection file *if we wrote it*

        Will not raise if the connection file was already removed somehow.
        """
        if self._connection_file_written:
            # cleanup connection files on full shutdown of kernel we started
            self._connection_file_written = False
            try:
                os.remove(self.connection_file)
            except (IOError, OSError, AttributeError):
                pass

    def cleanup_ipc_files(self):
        """Cleanup ipc files if we wrote them."""
        if self.transport != 'ipc':
            return
        for port in self.ports:
            ipcfile = "%s-%i" % (self.ip, port)
            try:
                os.remove(ipcfile)
            except (IOError, OSError):
                pass

    def _record_random_port_names(self):
        """Records which of the ports are randomly assigned.

        Records on first invocation, if the transport is tcp.
        Does nothing on later invocations."""

        if self.transport != 'tcp':
            return
        if self._random_port_names is not None:
            return

        self._random_port_names = []
        for name in port_names:
            if getattr(self, name) <= 0:
                self._random_port_names.append(name)

    def cleanup_random_ports(self):
        """Forgets randomly assigned port numbers and cleans up the connection file.

        Does nothing if no port numbers have been randomly assigned.
        In particular, does nothing unless the transport is tcp.
        """

        if not self._random_port_names:
            return

        for name in self._random_port_names:
            setattr(self, name, 0)

        self.cleanup_connection_file()

    def write_connection_file(self):
        """Write connection info to JSON dict in self.connection_file."""
        if self._connection_file_written and os.path.exists(self.connection_file):
            return

        self.connection_file, cfg = write_connection_file(self.connection_file,
            transport=self.transport, ip=self.ip, key=self.session.key,
            stdin_port=self.stdin_port, iopub_port=self.iopub_port,
            shell_port=self.shell_port, hb_port=self.hb_port,
            control_port=self.control_port,
            signature_scheme=self.session.signature_scheme,
            kernel_name=self.kernel_name
        )
        # write_connection_file also sets default ports:
        self._record_random_port_names()
        for name in port_names:
            setattr(self, name, cfg[name])

        self._connection_file_written = True

    def load_connection_file(self, connection_file=None):
        """Load connection info from JSON dict in self.connection_file.

        Parameters
        ----------
        connection_file: unicode, optional
            Path to connection file to load.
            If unspecified, use self.connection_file
        """
        if connection_file is None:
            connection_file = self.connection_file
        self.log.debug(u"Loading connection file %s", connection_file)
        with open(connection_file) as f:
            info = json.load(f)
        self.load_connection_info(info)

    def load_connection_info(self, info):
        """Load connection info from a dict containing connection info.

        Typically this data comes from a connection file
        and is called by load_connection_file.

        Parameters
        ----------
        info: dict
            Dictionary containing connection_info.
            See the connection_file spec for details.
        """
        self.transport = info.get('transport', self.transport)
        self.ip = info.get('ip', self._ip_default())

        self._record_random_port_names()
        for name in port_names:
            if getattr(self, name) == 0 and name in info:
                # not overridden by config or cl_args
                setattr(self, name, info[name])

        if 'key' in info:
            self.session.key = cast_bytes(info['key'])
        if 'signature_scheme' in info:
            self.session.signature_scheme = info['signature_scheme']

    #--------------------------------------------------------------------------
    # Creating connected sockets
    #--------------------------------------------------------------------------

    def _make_url(self, channel):
        """Make a ZeroMQ URL for a given channel."""
        transport = self.transport
        ip = self.ip
        port = getattr(self, '%s_port' % channel)

        if transport == 'tcp':
            return "tcp://%s:%i" % (ip, port)
        else:
            return "%s://%s-%s" % (transport, ip, port)

    def _create_connected_socket(self, channel, identity=None):
        """Create a zmq Socket and connect it to the kernel."""
        url = self._make_url(channel)
        socket_type = channel_socket_types[channel]
        self.log.debug("Connecting to: %s" % url)
        sock = self.context.socket(socket_type)
        # set linger to 1s to prevent hangs at exit
        sock.linger = 1000
        if identity:
            sock.identity = identity
        sock.connect(url)
        return sock

    def connect_iopub(self, identity=None):
        """return zmq Socket connected to the IOPub channel"""
        sock = self._create_connected_socket('iopub', identity=identity)
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        return sock

    def connect_shell(self, identity=None):
        """return zmq Socket connected to the Shell channel"""
        return self._create_connected_socket('shell', identity=identity)

    def connect_stdin(self, identity=None):
        """return zmq Socket connected to the StdIn channel"""
        return self._create_connected_socket('stdin', identity=identity)

    def connect_hb(self, identity=None):
        """return zmq Socket connected to the Heartbeat channel"""
        return self._create_connected_socket('hb', identity=identity)

    def connect_control(self, identity=None):
        """return zmq Socket connected to the Control channel"""
        return self._create_connected_socket('control', identity=identity)


__all__ = [
    'write_connection_file',
    'find_connection_file',
    'tunnel_to_kernel',
]
