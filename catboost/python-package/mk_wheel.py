from __future__ import print_function

import subprocess
import shutil
import os
import sys
import platform
import tempfile


sys.dont_write_bytecode = True

PL_LINUX = 'manylinux1_x86_64'
PL_MACOS = 'macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64'
PL_WIN = 'win_amd64'


class PythonVersion(object):
    def __init__(self, major, minor, from_sandbox=False):
        self.major = major
        self.minor = minor
        self.from_sandbox = from_sandbox


class PythonTrait(object):
    def __init__(self, arc_root, out_root, tail_args):
        self.arc_root = arc_root
        self.out_root = out_root
        self.tail_args = tail_args
        self.python_version = mine_system_python_ver(self.tail_args)
        self.platform = mine_platform(self.tail_args)
        self.py_config, self.lang = self.get_python_info()

    def gen_cmd(self):
        cmd = [
            sys.executable, arc_root + '/ya', 'make', os.path.join(arc_root, 'catboost', 'python-package', 'catboost'),
            '--no-src-links', '-r', '--output', out_root, '-DPYTHON_CONFIG=' + self.py_config, '-DNO_DEBUGINFO', '-DOS_SDK=local',
        ]

        if not self.python_version.from_sandbox:
            cmd += ['-DUSE_ARCADIA_PYTHON=no']
            cmd += extra_opts(self._on_win())

        cmd += self.tail_args
        return cmd

    def get_python_info(self):
        if self.python_version.major == 2:
            py_config = 'python-config'
            lang = 'cp27'
        else:
            py_config = 'python3-config'
            lang = 'cp3' + str(self.python_version.minor)
        return py_config, lang

    def so_name(self):
        if self._on_win():
            return '_catboost.pyd'

        return '_catboost.so'

    def dll_ext(self):
        if self._on_win():
            return '.pyd'
        return '.so'

    def _on_win(self):
        if self.platform == PL_WIN:
            return True
        return platform.system() == 'Windows'


def mine_platform(tail_args):
    platform = find_target_platform(tail_args)
    if platform:
        return transform_platform(platform)
    return gen_platform()


def gen_platform():
    import distutils.util

    value = distutils.util.get_platform().replace("linux", "manylinux1")
    value = value.replace('-', '_').replace('.', '_')
    if 'macosx' in value:
        value = PL_MACOS
    return value


def find_target_platform(tail_args):
    try:
        target_platform_index = tail_args.index('--target-platform')
        return tail_args[target_platform_index + 1].lower()
    except ValueError:
        target_platform = [arg for arg in tail_args if '--target-platform' in arg]
        if target_platform:
            _, platform = target_platform[0].split('=')
            return platform.lower()
    return None


def transform_platform(platform):
    if 'linux' in platform:
        return PL_LINUX
    elif 'darwin' in platform:
        return PL_MACOS
    elif 'win' in platform:
        return PL_WIN
    else:
        raise Exception('Unsupported platform {}'.format(platform))


def get_version():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    version_py = os.path.join(CURRENT_DIR, 'version.py')
    exec(compile(open(version_py, "rb").read(), version_py, 'exec'))

    return locals()['VERSION']


def extra_opts(on_win=False):
    if on_win:
        py_dir = os.path.dirname(sys.executable)
        include_path = os.path.join(py_dir, 'include')
        py_libs = os.path.join(py_dir, 'libs', 'python{}{}.lib'.format(sys.version_info.major, sys.version_info.minor))
        return ['-DPYTHON_INCLUDE=/I ' + include_path, '-DPYTHON_LIBRARIES=' + py_libs]

    return []


def find_info_in_args(tail_args):
    def prepare_info(arg):
        _, version = arg.split('=')
        major, minor = version.split('.')
        py_config = 'python-config' if major == '2' else 'python3-config'
        lang = 'cp{major}{minor}'.format(major=major, minor=minor)
        return py_config, lang

    for arg in tail_args:
        if 'USE_SYSTEM_PYTHON' in arg:
            return prepare_info(arg)

    return None, None


def mine_system_python_ver(tail_args):
    for arg in tail_args:
        if 'USE_SYSTEM_PYTHON' in arg:
            _, version = arg.split('=')
            major, minor = version.split('.')
            return PythonVersion(int(major), int(minor), from_sandbox=True)
    return PythonVersion(sys.version_info.major, sys.version_info.minor)


def build(arc_root, out_root, tail_args):
    os.chdir(os.path.join(arc_root, 'catboost', 'python-package', 'catboost'))

    py_trait = PythonTrait(arc_root, out_root, tail_args)
    ver = get_version()

    shutil.rmtree('catboost', ignore_errors=True)
    os.makedirs('catboost/catboost')
    try:
        print('Trying to build GPU version', file=sys.stderr)
        gpu_cmd = py_trait.gen_cmd() + ['-DHAVE_CUDA=yes']
        print(' '.join(gpu_cmd), file=sys.stderr)
        subprocess.check_call(gpu_cmd)
        print('Build GPU version: OK', file=sys.stderr)
        os.makedirs('catboost/catboost/gpu')
        open('catboost/catboost/gpu/__init__.py', 'w').close()
        shutil.copy(os.path.join(py_trait.out_root, 'catboost', 'python-package', 'catboost', py_trait.so_name()), 'catboost/catboost/gpu/_catboost' + py_trait.dll_ext())
    except Exception:
        print('GPU version build failed', file=sys.stderr)

    print('Building CPU version', file=sys.stderr)
    cpu_cmd = py_trait.gen_cmd() + ['-DHAVE_CUDA=no']
    print(' '.join(cpu_cmd), file=sys.stderr)
    subprocess.check_call(cpu_cmd)
    print('Building CPU version: OK', file=sys.stderr)
    shutil.copy(os.path.join(py_trait.out_root, 'catboost', 'python-package', 'catboost', py_trait.so_name()), 'catboost/catboost/_catboost' + py_trait.dll_ext())

    shutil.copy('__init__.py', 'catboost/catboost/__init__.py')
    shutil.copy('version.py', 'catboost/catboost/version.py')
    shutil.copy('core.py', 'catboost/catboost/core.py')
    shutil.copy('datasets.py', 'catboost/catboost/datasets.py')
    shutil.copy('utils.py', 'catboost/catboost/utils.py')
    shutil.copytree('eval', 'catboost/catboost/eval')
    shutil.copytree('widget', 'catboost/catboost/widget')
    dist_info_dir = 'catboost/catboost-{}.dist-info'.format(ver)
    shutil.copytree(os.path.join(py_trait.arc_root, 'catboost', 'python-package', 'catboost.dist-info'), dist_info_dir)

    with open(os.path.join(dist_info_dir, 'METADATA'), 'r') as fm:
        metadata = fm.read()
    metadata = metadata.format(version=ver)
    with open(os.path.join(dist_info_dir, 'METADATA'), 'w') as fm:
        fm.write(metadata)

    wheel_name = 'catboost-{}-{}-none-{}.whl'.format(ver, py_trait.lang, py_trait.platform)

    try:
        os.remove(wheel_name)
    except OSError:
        pass

    shutil.make_archive(wheel_name, 'zip', 'catboost')
    os.rename(wheel_name + '.zip', wheel_name)
    shutil.move(wheel_name, os.path.join(py_trait.arc_root, 'catboost', 'python-package', wheel_name))

    return wheel_name


if __name__ == '__main__':
    arc_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    out_root = tempfile.mkdtemp()
    wheel_name = build(arc_root, out_root, sys.argv[1:])
    print(wheel_name)
