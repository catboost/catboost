from __future__ import print_function

import subprocess
import shutil
import os
import sys
import platform
import tempfile


sys.dont_write_bytecode = True


def gen_platform():
    import distutils.util

    value = distutils.util.get_platform().replace("linux", "manylinux1")
    value = value.replace('-', '_').replace('.', '_')
    return value


def on_win():
    return platform.system() == 'Windows'


def get_version():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    version_py = os.path.join(CURRENT_DIR, 'version.py')
    exec(compile(open(version_py, "rb").read(), version_py, 'exec'))

    return locals()['VERSION']


def extra_opts():
    if on_win():
        py_dir = os.path.dirname(sys.executable)
        include_path = os.path.join(py_dir, 'include')
        py_libs = os.path.join(py_dir, 'libs', 'python{}{}.lib'.format(sys.version_info.major, sys.version_info.minor))
        return ['-DPYTHON_INCLUDE=/I ' + include_path, '-DPYTHON_LIBRARIES=' + py_libs]

    return []


def dll_ext():
    if on_win():
        return '.pyd'
    return '.so'


def so_name():
    if on_win():
        return '_catboost.pyd'

    return '_catboost.so'


def build(arc_root, out_root, tail_args):
    os.chdir(os.path.join(arc_root, 'catboost', 'python-package', 'catboost'))

    if sys.version_info.major == 2:
        py_config = 'python-config'
        lang = 'cp27'
    else:
        py_config = 'python3-config'
        lang = 'py3'
    ver = get_version()
    cmd = [
              sys.executable, arc_root + '/ya', 'make', os.path.join(arc_root, 'catboost', 'python-package', 'catboost'),
              '--no-src-links', '-r', '--output', out_root, '-DUSE_ARCADIA_PYTHON=no', '-DPYTHON_CONFIG=' + py_config
          ] + extra_opts() + tail_args
    print(' '.join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)

    shutil.rmtree('catboost', ignore_errors=True)
    os.makedirs('catboost/catboost')
    shutil.copy(os.path.join(out_root, 'catboost', 'python-package', 'catboost', so_name()), 'catboost/catboost/_catboost' + dll_ext())
    shutil.copy('__init__.py', 'catboost/catboost/__init__.py')
    shutil.copy('version.py', 'catboost/catboost/version.py')
    shutil.copy('core.py', 'catboost/catboost/core.py')
    shutil.copytree('widget', 'catboost/catboost/widget')
    shutil.copytree(os.path.join(arc_root, 'catboost', 'python-package','catboost.dist-info'), 'catboost/catboost-{}.dist-info'.format(ver))

    plat = gen_platform()
    wheel_name = 'catboost-{}-{}-none-{}.whl'.format(ver, lang, plat)

    try:
        os.remove(wheel_name)
    except OSError:
        pass

    shutil.make_archive(wheel_name, 'zip', 'catboost')
    os.rename(wheel_name + '.zip', wheel_name)
    shutil.move(wheel_name, os.path.join(arc_root, 'catboost', 'python-package'))

    return wheel_name


if __name__ == '__main__':
    arc_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    out_root = tempfile.mkdtemp()
    wheel_name = build(arc_root, out_root, sys.argv[1:])
    print(wheel_name)
