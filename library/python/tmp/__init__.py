import logging
import tempfile
import time
import os
import shutil
import contextlib

import library.python.fs as fs
import library.python.unique_id as uniq_id

logger = logging.getLogger(__name__)


_tmp_roots = []
_startup_tmp_dir = None
_startup_tmp_dir_set = False


def set_tmp_dir(root, keep_dir):
    uniq_name = '{0}.{1}.{2}'.format(int(time.time()), os.getpid(), uniq_id.gen8())
    tmp_dir = os.path.join(root, uniq_name)
    fs.create_dirs(tmp_dir)

    old_tmp_dir = os.environ.get('TMPDIR')
    logger.debug('Set TMPDIR=%s instead of %s', tmp_dir, old_tmp_dir)
    os.environ['TMPDIR'] = tmp_dir

    global _startup_tmp_dir
    global _startup_tmp_dir_set
    if not _startup_tmp_dir_set:
        _startup_tmp_dir = old_tmp_dir
        _startup_tmp_dir_set = True

    if not keep_dir:
        _tmp_roots.append(tmp_dir)


def remove_tmp_dirs(env=None):
    if env is None:
        env = os.environ

    global _tmp_roots

    for x in _tmp_roots:
        logger.debug('Removing tmp dir %s', x)
        shutil.rmtree(x, ignore_errors=True)

    _tmp_roots = []

    revert_tmp_dir(env)


def revert_tmp_dir(env=None):
    if not env:
        env = os.environ

    if _startup_tmp_dir_set:
        if _startup_tmp_dir is not None:
            env['TMPDIR'] = _startup_tmp_dir
        else:
            env.pop('TMPDIR', None)
        logger.debug('Reset back TMPDIR=%s', env.get('TMPDIR'))


def temp_path(path):
    class Remover(object):
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self.path

        def __exit__(self, exc_type, exc_value, traceback):
            if _tmp_roots or not _startup_tmp_dir_set:
                fs.remove_tree_safe(path)

    return Remover(path)


def create_temp_file(prefix='yatmp'):
    fd, path = tempfile.mkstemp(prefix=prefix)
    os.close(fd)
    return path


def temp_dir(prefix='yatmp', dir=None):
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=dir)
    return temp_path(tmpdir)


def temp_file(prefix='yatmp'):
    tmpfile = create_temp_file(prefix=prefix)
    return temp_path(tmpfile)


@contextlib.contextmanager
def environment(env):

    def set_env(e):
        os.environ.clear()
        for k, v in e.items():
            os.environ[k] = v

    stored = os.environ.copy()
    try:
        set_env(env or {})
        yield
    finally:
        set_env(stored)
