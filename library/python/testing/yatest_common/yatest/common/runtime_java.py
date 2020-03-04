import os
import tarfile
import contextlib


def get_java_path(jdk_dir):
    # deprecated - to be deleted
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


def get_build_java_dir(jdk_dir):
    versions = [8, 10, 11, 12, 13]

    for version in versions:
        jdk_tar_path = os.path.join(jdk_dir, "jdk{}.tar".format(version))
        if os.path.exists(jdk_tar_path):
            with contextlib.closing(tarfile.open(jdk_tar_path)) as tf:
                tf.extractall(jdk_dir)
            assert os.path.exists(os.path.join(jdk_dir, "bin", "java"))
            return jdk_dir
    return None
