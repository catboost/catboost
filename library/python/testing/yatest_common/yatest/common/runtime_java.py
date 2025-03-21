import os
import re
import tarfile
import contextlib

from . import runtime

_JAVA_DIR = []
_JDK_RE = re.compile(r'jdk(\d+)\.tar')


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
    if _JAVA_DIR:
        return _JAVA_DIR[0]

    jdk_dest_dir = runtime.build_path('jdk4test')
    if os.path.exists(os.path.join(jdk_dest_dir, "bin", "java")):
        _JAVA_DIR.append(jdk_dest_dir)
        return jdk_dest_dir

    # Get jdk with the lowest version
    jdk_tar_version = None
    jdk_tar_path = None
    for name in os.listdir(jdk_dir):
        m = _JDK_RE.match(name)
        if m:
            v = int(m.group(1))
            if jdk_tar_version is None or jdk_tar_version > v:
                jdk_tar_version = v
                jdk_tar_path = os.path.join(jdk_dir, name)

    if jdk_tar_path is None:
        _JAVA_DIR.append(None)
        return None

    with contextlib.closing(tarfile.open(jdk_tar_path)) as tf:
        tf.extractall(jdk_dest_dir)
    assert os.path.exists(os.path.join(jdk_dest_dir, "bin", "java"))
    _JAVA_DIR.append(jdk_dest_dir)
    return jdk_dest_dir
