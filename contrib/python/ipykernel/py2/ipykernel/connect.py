"""Connection file-related utilities for the kernel
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import absolute_import

import json
import sys
from subprocess import Popen, PIPE
import warnings

from IPython.core.profiledir import ProfileDir
from IPython.paths import get_ipython_dir
from ipython_genutils.path import filefind
from ipython_genutils.py3compat import str_to_bytes, PY3

import jupyter_client
from jupyter_client import write_connection_file



def get_connection_file(app=None):
    """Return the path to the connection file of an app

    Parameters
    ----------
    app : IPKernelApp instance [optional]
        If unspecified, the currently running app will be used
    """
    if app is None:
        from ipykernel.kernelapp import IPKernelApp
        if not IPKernelApp.initialized():
            raise RuntimeError("app not specified, and not in a running Kernel")

        app = IPKernelApp.instance()
    return filefind(app.connection_file, ['.', app.connection_dir])


def find_connection_file(filename='kernel-*.json', profile=None):
    """DEPRECATED: find a connection file, and return its absolute path.
    
    THIS FUNCION IS DEPRECATED. Use juptyer_client.find_connection_file instead.

    Parameters
    ----------
    filename : str
        The connection file or fileglob to search for.
    profile : str [optional]
        The name of the profile to use when searching for the connection file,
        if different from the current IPython session or 'default'.

    Returns
    -------
    str : The absolute path of the connection file.
    """
    
    import warnings
    warnings.warn("""ipykernel.find_connection_file is deprecated, use jupyter_client.find_connection_file""",
        DeprecationWarning, stacklevel=2)
    from IPython.core.application import BaseIPythonApplication as IPApp
    try:
        # quick check for absolute path, before going through logic
        return filefind(filename)
    except IOError:
        pass

    if profile is None:
        # profile unspecified, check if running from an IPython app
        if IPApp.initialized():
            app = IPApp.instance()
            profile_dir = app.profile_dir
        else:
            # not running in IPython, use default profile
            profile_dir = ProfileDir.find_profile_dir_by_name(get_ipython_dir(), 'default')
    else:
        # find profiledir by profile name:
        profile_dir = ProfileDir.find_profile_dir_by_name(get_ipython_dir(), profile)
    security_dir = profile_dir.security_dir
    
    return jupyter_client.find_connection_file(filename, path=['.', security_dir])


def _find_connection_file(connection_file, profile=None):
    """Return the absolute path for a connection file
    
    - If nothing specified, return current Kernel's connection file
    - If profile specified, show deprecation warning about finding connection files in profiles
    - Otherwise, call jupyter_client.find_connection_file
    """
    if connection_file is None:
        # get connection file from current kernel
        return get_connection_file()
    else:
        # connection file specified, allow shortnames:
        if profile is not None:
            warnings.warn(
                "Finding connection file by profile is deprecated.",
                DeprecationWarning, stacklevel=3,
            )
            return find_connection_file(connection_file, profile=profile)
        else:
            return jupyter_client.find_connection_file(connection_file)


def get_connection_info(connection_file=None, unpack=False, profile=None):
    """Return the connection information for the current Kernel.

    Parameters
    ----------
    connection_file : str [optional]
        The connection file to be used. Can be given by absolute path, or
        IPython will search in the security directory of a given profile.
        If run from IPython,

        If unspecified, the connection file for the currently running
        IPython Kernel will be used, which is only allowed from inside a kernel.
    unpack : bool [default: False]
        if True, return the unpacked dict, otherwise just the string contents
        of the file.
    profile : DEPRECATED

    Returns
    -------
    The connection dictionary of the current kernel, as string or dict,
    depending on `unpack`.
    """
    cf = _find_connection_file(connection_file, profile)

    with open(cf) as f:
        info = f.read()

    if unpack:
        info = json.loads(info)
        # ensure key is bytes:
        info['key'] = str_to_bytes(info.get('key', ''))
    return info


def connect_qtconsole(connection_file=None, argv=None, profile=None):
    """Connect a qtconsole to the current kernel.

    This is useful for connecting a second qtconsole to a kernel, or to a
    local notebook.

    Parameters
    ----------
    connection_file : str [optional]
        The connection file to be used. Can be given by absolute path, or
        IPython will search in the security directory of a given profile.
        If run from IPython,

        If unspecified, the connection file for the currently running
        IPython Kernel will be used, which is only allowed from inside a kernel.
    argv : list [optional]
        Any extra args to be passed to the console.
    profile : DEPRECATED

    Returns
    -------
    :class:`subprocess.Popen` instance running the qtconsole frontend
    """
    argv = [] if argv is None else argv

    cf = _find_connection_file(connection_file, profile)

    cmd = ';'.join([
        "from IPython.qt.console import qtconsoleapp",
        "qtconsoleapp.main()"
    ])

    kwargs = {}
    if PY3:
        # Launch the Qt console in a separate session & process group, so
        # interrupting the kernel doesn't kill it. This kwarg is not on Py2.
        kwargs['start_new_session'] = True

    return Popen([sys.executable, '-c', cmd, '--existing', cf] + argv,
        stdout=PIPE, stderr=PIPE, close_fds=(sys.platform != 'win32'),
        **kwargs
    )


__all__ = [
    'write_connection_file',
    'get_connection_file',
    'find_connection_file',
    'get_connection_info',
    'connect_qtconsole',
]
