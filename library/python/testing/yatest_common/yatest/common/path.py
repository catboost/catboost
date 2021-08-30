# coding=utf-8

import errno
import os
import shutil
import contextlib

import library.python.fs as lpf


def replace_in_file(path, old, new):
    """
    Replace text occurrences in a file
    :param path: path to the file
    :param old: text to replace
    :param new: replacement
    """
    with open(path) as fp:
        content = fp.read()

    lpf.ensure_removed(path)
    with open(path, 'w') as fp:
        fp.write(content.replace(old, new))


@contextlib.contextmanager
def change_dir(path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(old)


def copytree(src, dst, symlinks=False, ignore=None, postprocessing=None):
    '''
    Copy an entire directory of files into an existing directory
    instead of raising Exception what shtuil.copytree does
    '''
    if not os.path.exists(dst) and os.path.isdir(src):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
    if postprocessing:
        postprocessing(dst, False)
        for root, dirs, files in os.walk(dst):
            for path in dirs:
                postprocessing(os.path.join(root, path), False)
            for path in files:
                postprocessing(os.path.join(root, path), True)


def get_unique_file_path(dir_path, file_pattern, create_file=True, max_suffix=10000):
    def atomic_file_create(path):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            return True
        except OSError as e:
            if e.errno in [errno.EEXIST, errno.EISDIR, errno.ETXTBSY]:
                return False
            # Access issue with file itself, not parent directory.
            if e.errno == errno.EACCES and os.path.exists(path):
                return False
            raise e

    def atomic_dir_create(path):
        try:
            os.mkdir(path)
            return True
        except OSError as e:
            if e.errno == errno.EEXIST:
                return False
            raise e

    file_path = os.path.join(dir_path, file_pattern)
    lpf.ensure_dir(os.path.dirname(file_path))
    file_counter = 0
    handler = atomic_file_create if create_file else atomic_dir_create
    while os.path.exists(file_path) or not handler(file_path):
        file_path = os.path.join(dir_path, file_pattern + ".{}".format(file_counter))
        file_counter += 1
        assert file_counter < max_suffix
    return file_path
