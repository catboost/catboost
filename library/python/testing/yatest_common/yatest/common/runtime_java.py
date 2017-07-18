import os
import tarfile
import contextlib


def get_java_path(jdk_dir):
    java_paths = (os.path.join(jdk_dir, 'bin', 'java'), os.path.join(jdk_dir, 'bin', 'java.exe'))

    for p in java_paths:
        if os.path.exists(p):
            return p

    for f in os.listdir(jdk_dir):
        if f.endswith('.tar'):
            with contextlib.closing(tarfile.open(os.path.join(jdk_dir, f))) as tf:
                tf.extractall(jdk_dir)

    for p in java_paths:
        if os.path.exists(p):
            return p

    return ''
