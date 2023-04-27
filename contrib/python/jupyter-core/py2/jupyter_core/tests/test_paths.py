"""Tests for paths"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import re
import stat
import shutil
import tempfile

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch # py2

from jupyter_core import paths
from jupyter_core.paths import (
    jupyter_config_dir, jupyter_data_dir, jupyter_runtime_dir,
    jupyter_path, ENV_JUPYTER_PATH,
    secure_write, is_hidden, is_file_hidden
)
from ipython_genutils.tempdir import TemporaryDirectory
from ipython_genutils.py3compat import cast_unicode
from .mocking import darwin, windows, linux

import pytest

pjoin = os.path.join


xdg_env = {
    'XDG_CONFIG_HOME': '/tmp/xdg/config',
    'XDG_DATA_HOME': '/tmp/xdg/data',
    'XDG_RUNTIME_DIR': '/tmp/xdg/runtime',
}
xdg = patch.dict('os.environ', xdg_env)
no_xdg = patch.dict('os.environ', {
    'XDG_CONFIG_HOME': '',
    'XDG_DATA_HOME': '',
    'XDG_RUNTIME_DIR': '',
})

appdata = patch.dict('os.environ', {'APPDATA': 'appdata'})

no_config_env = patch.dict('os.environ', {
    'JUPYTER_CONFIG_DIR': '',
    'JUPYTER_DATA_DIR': '',
    'JUPYTER_RUNTIME_DIR': '',
    'JUPYTER_PATH': '',
})

jupyter_config_env = '/jupyter-cfg'
config_env = patch.dict('os.environ', {'JUPYTER_CONFIG_DIR': jupyter_config_env})


def realpath(path):
    return os.path.realpath(os.path.expanduser(path))

home_jupyter = realpath('~/.jupyter')


def test_config_dir_darwin():
    with darwin, no_config_env:
        config = jupyter_config_dir()
    assert config == home_jupyter

    with darwin, config_env:
        config = jupyter_config_dir()
    assert config == jupyter_config_env


def test_config_dir_windows():
    with windows, no_config_env:
        config = jupyter_config_dir()
    assert config == home_jupyter

    with windows, config_env:
        config = jupyter_config_dir()
    assert config == jupyter_config_env


def test_config_dir_linux():
    with windows, no_config_env:
        config = jupyter_config_dir()
    assert config == home_jupyter

    with windows, config_env:
        config = jupyter_config_dir()
    assert config == jupyter_config_env


def test_data_dir_env():
    data_env = 'runtime-dir'
    with patch.dict('os.environ', {'JUPYTER_DATA_DIR': data_env}):
        data = jupyter_data_dir()
    assert data == data_env


def test_data_dir_darwin():
    with darwin:
        data = jupyter_data_dir()
    assert data == realpath('~/Library/Jupyter')

    with darwin, xdg:
        # darwin should ignore xdg
        data = jupyter_data_dir()
    assert data == realpath('~/Library/Jupyter')


def test_data_dir_windows():
    with windows, appdata:
        data = jupyter_data_dir()
    assert data == pjoin('appdata', 'jupyter')

    with windows, appdata, xdg:
        # windows should ignore xdg
        data = jupyter_data_dir()
    assert data == pjoin('appdata', 'jupyter')


def test_data_dir_linux():
    with linux, no_xdg:
        data = jupyter_data_dir()
    assert data == realpath('~/.local/share/jupyter')

    with linux, xdg:
        data = jupyter_data_dir()
    assert data == pjoin(xdg_env['XDG_DATA_HOME'], 'jupyter')


def test_runtime_dir_env():
    rtd_env = 'runtime-dir'
    with patch.dict('os.environ', {'JUPYTER_RUNTIME_DIR': rtd_env}):
        runtime = jupyter_runtime_dir()
    assert runtime == rtd_env


def test_runtime_dir_darwin():
    with darwin:
        runtime = jupyter_runtime_dir()
    assert runtime == realpath('~/Library/Jupyter/runtime')

    with darwin, xdg:
        # darwin should ignore xdg
        runtime = jupyter_runtime_dir()
    assert runtime == realpath('~/Library/Jupyter/runtime')


def test_runtime_dir_windows():
    with windows, appdata:
        runtime = jupyter_runtime_dir()
    assert runtime == pjoin('appdata', 'jupyter', 'runtime')

    with windows, appdata, xdg:
        # windows should ignore xdg
        runtime = jupyter_runtime_dir()
    assert runtime == pjoin('appdata', 'jupyter', 'runtime')


def test_runtime_dir_linux():
    with linux, no_xdg:
        runtime = jupyter_runtime_dir()
    assert runtime == realpath('~/.local/share/jupyter/runtime')

    with linux, xdg:
        runtime = jupyter_runtime_dir()
    assert runtime == pjoin(xdg_env['XDG_DATA_HOME'], 'jupyter', 'runtime')


def test_jupyter_path():
    system_path = ['system', 'path']
    with no_config_env, patch.object(paths, 'SYSTEM_JUPYTER_PATH', system_path):
        path = jupyter_path()
    assert path[0] == jupyter_data_dir()
    assert path[-2:] == system_path


def test_jupyter_path_env():
    path_env = os.pathsep.join([
        pjoin('foo', 'bar'),
        pjoin('bar', 'baz', ''), # trailing /
    ])

    with patch.dict('os.environ', {'JUPYTER_PATH': path_env}):
        path = jupyter_path()
    assert path[:2] == [pjoin('foo', 'bar'), pjoin('bar', 'baz')]


def test_jupyter_path_sys_prefix():
    with patch.object(paths, 'ENV_JUPYTER_PATH', ['sys_prefix']):
        path = jupyter_path()
    assert 'sys_prefix' in path


def test_jupyter_path_subdir():
    path = jupyter_path('sub1', 'sub2')
    for p in path:
        assert p.endswith(pjoin('', 'sub1', 'sub2'))


def test_is_hidden():
    with TemporaryDirectory() as root:
        subdir1 = os.path.join(root, 'subdir')
        os.makedirs(subdir1)
        assert not is_hidden(subdir1, root)
        assert not is_file_hidden(subdir1)

        subdir2 = os.path.join(root, '.subdir2')
        os.makedirs(subdir2)
        assert is_hidden(subdir2, root)
        assert is_file_hidden(subdir2)
        # root dir is always visible
        assert not is_hidden(subdir2, subdir2)

        subdir34 = os.path.join(root, 'subdir3', '.subdir4')
        os.makedirs(subdir34)
        assert is_hidden(subdir34, root)
        assert is_hidden(subdir34)

        subdir56 = os.path.join(root, '.subdir5', 'subdir6')
        os.makedirs(subdir56)
        assert is_hidden(subdir56, root)
        assert is_hidden(subdir56)
        assert not is_file_hidden(subdir56)
        assert not is_file_hidden(subdir56, os.stat(subdir56))


@pytest.mark.skipif("sys.platform != 'win32'")
def test_is_hidden_win32():
    import ctypes
    with TemporaryDirectory() as root:
        root = cast_unicode(root)
        subdir1 = os.path.join(root, u'subdir')
        os.makedirs(subdir1)
        assert not is_hidden(subdir1, root)
        r = ctypes.windll.kernel32.SetFileAttributesW(subdir1, 0x02)
        print(r) # Helps debugging
        assert is_hidden(subdir1, root)
        assert is_file_hidden(subdir1)


@pytest.mark.skipif("sys.platform != 'win32'")
def test_secure_write_win32():
    def fetch_win32_permissions(filename):
        '''Extracts file permissions on windows using icacls'''
        role_permissions = {}
        for index, line in enumerate(os.popen("icacls %s" % filename).read().splitlines()):
            if index == 0:
                line = line.split(filename)[-1].strip().lower()
            match = re.match(r'\s*([^:]+):\(([^\)]*)\)', line)
            if match:
                usergroup, permissions = match.groups()
                usergroup = usergroup.lower().split('\\')[-1]
                permissions = set(p.lower() for p in permissions.split(','))
                role_permissions[usergroup] = permissions
            elif not line.strip():
                break
        return role_permissions

    def check_user_only_permissions(fname):
        # Windows has it's own permissions ACL patterns
        import win32api
        username = win32api.GetUserName().lower()
        permissions = fetch_win32_permissions(fname)
        print(permissions) # for easier debugging
        assert username in permissions
        assert permissions[username] == set(['r', 'w'])
        assert 'administrators' in permissions
        assert permissions['administrators'] == set(['f'])
        assert 'everyone' not in permissions
        assert len(permissions) == 2

    directory = tempfile.mkdtemp()
    fname = os.path.join(directory, 'check_perms')
    try:
        with secure_write(fname) as f:
            f.write('test 1')
        check_user_only_permissions(fname)
        with open(fname, 'r') as f:
            assert f.read() == 'test 1'
    finally:
        shutil.rmtree(directory)


@pytest.mark.skipif("sys.platform == 'win32'")
def test_secure_write_unix():
    directory = tempfile.mkdtemp()
    fname = os.path.join(directory, 'check_perms')
    try:
        with secure_write(fname) as f:
            f.write('test 1')
        mode = os.stat(fname).st_mode
        assert 0o0600 == (stat.S_IMODE(mode) & 0o7677)  # tolerate owner-execute bit
        with open(fname, 'r') as f:
            assert f.read() == 'test 1'

        # Try changing file permissions ahead of time
        os.chmod(fname, 0o755)
        with secure_write(fname) as f:
            f.write('test 2')
        mode = os.stat(fname).st_mode
        assert 0o0600 == (stat.S_IMODE(mode) & 0o7677)  # tolerate owner-execute bit
        with open(fname, 'r') as f:
            assert f.read() == 'test 2'
    finally:
        shutil.rmtree(directory)
