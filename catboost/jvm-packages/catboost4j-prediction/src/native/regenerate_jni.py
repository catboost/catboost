#!/usr/bin/env python
#
# Regenerate JNI headers for
#
# NOTE: doesn't work with JDK 10 because javah was removed [1] from JDK, and javac doesn't seem to
# be able to generate native headers from .class files.
#
# NOTE: this script must be python2/3 compatible

from __future__ import absolute_import, print_function

import distutils.spawn
import os
import subprocess
import sys


def _get_arcadia_root():
    arcadia_root = None
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    while True:
        if os.path.isfile(os.path.join(path, '.arcadia.root')):
            arcadia_root = path
            break

        if path == os.path.dirname(path):
            break

        path = os.path.dirname(path)

    assert arcadia_root is not None, 'you are probably trying to use this script with repository being checkout not from the root'
    return arcadia_root


def _get_native_lib_dir(relative=None):
    if relative is None:
        relative = _get_arcadia_root()
    return os.path.join(
        relative,
        os.path.join(*'catboost/jvm-packages/catboost4j-prediction/src/native'.split('/')))


def _get_classes_dir():
    return os.path.join(
        _get_arcadia_root(),
        os.path.join(*'catboost/jvm-packages/catboost4j-prediction/target/classes'.split('/')))


def _run_javah(args, env=None):
    if env is None:
        env = os.environ.copy()
    java_home = env.get('JAVA_HOME')
    if java_home is not None:
        javah_path = os.path.join(java_home, os.path.join(*'bin/javah'.split('/')))
        subprocess.check_call(
            [javah_path] + args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr)
        return

    distutils.spawn.spawn(['javah'] + args)


def _fix_header(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    if not data.startswith(b'#pragma once\n'):
        with open(filename, 'wb') as f:
            f.write(b'#pragma once\n\n')
            f.write(data)


def _main():
    javah_args = [
        '-verbose',
        '-d', _get_native_lib_dir(),
        '-jni',
        '-classpath', _get_classes_dir(),
        'ai.catboost.CatBoostJNIImpl']

    _run_javah(javah_args)
    _fix_header(os.path.join(
        _get_native_lib_dir(),
        'ai_catboost_CatBoostJNIImpl.h'))


if '__main__' == __name__:
    _main()
