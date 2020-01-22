# coding=utf-8

import os
import shutil
import contextlib


def replace_in_file(path, old, new):
    """
    Replace text occurrences in a file
    :param path: path to the file
    :param old: text to replace
    :param new: replacement
    """
    with open(path) as fp:
        content = fp.read()

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


def copytree(src, dst, symlinks=False, ignore=None):
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


def get_unique_file_path(dir_path, file_pattern):
    file_path = os.path.join(dir_path, file_pattern)
    file_counter = 0
    while os.path.exists(file_path):
        file_path = os.path.join(dir_path, file_pattern + ".{}".format(file_counter))
        file_counter += 1
    return file_path
