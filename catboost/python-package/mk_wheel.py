from __future__ import print_function

import subprocess
import shutil
import os
import stat
import sys
import platform
import tempfile
import hashlib

from base64 import urlsafe_b64encode


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


def get_version(version_py):
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


def allow_to_write(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IWRITE)


def calc_sha256_digest(filename):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            chunk_size = 65536
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.digest()


def make_record(dir_path, dist_info_dir):
    record_filename = os.path.join(dist_info_dir, 'RECORD')
    with open(record_filename, 'w') as record:
        wheel_items = []
        for root, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                wheel_items.append(os.path.join(root, filename))
        tmp_dir_length = len(dir_path) + 1
        for item in wheel_items:
            if item != record_filename:
                record.write(item[tmp_dir_length:] + ',sha256=' + urlsafe_b64encode(calc_sha256_digest(item)).decode('ascii') + ',' + str(os.path.getsize(item)) + '\n')
            else:
                record.write(item[tmp_dir_length:] + ',,\n')


def make_wheel(wheel_name, pkg_name, ver, arc_root, so_path):
    dir_path = tempfile.mkdtemp()

    # Create py files
    python_package_dir = os.path.join(arc_root, 'catboost/python-package')
    os.makedirs(os.path.join(dir_path, pkg_name))
    for file_name in ['__init__.py', 'version.py', 'core.py', 'datasets.py', 'utils.py', 'eval', 'widget', 'monoforest.py', 'text_processing.py']:
        src = os.path.join(python_package_dir, 'catboost', file_name)
        dst = os.path.join(dir_path, pkg_name, file_name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)

    # Create so files
    so_name = PythonTrait('', '', []).so_name()
    shutil.copy(so_path, os.path.join(dir_path, pkg_name, so_name))

    # Create metadata
    dist_info_dir = os.path.join(dir_path, '{}-{}.dist-info'.format(pkg_name, ver))
    shutil.copytree(os.path.join(python_package_dir, 'catboost.dist-info'), dist_info_dir)

    def substitute_vars(file_path):
        allow_to_write(file_path)
        with open(file_path, 'r') as fm:
            metadata = fm.read()
        metadata = metadata.format(
            pkg_name=pkg_name,
            version=ver,
        )
        with open(file_path, 'w') as fm:
            fm.write(metadata)
    substitute_vars(os.path.join(dist_info_dir, 'METADATA'))
    substitute_vars(os.path.join(dist_info_dir, 'top_level.txt'))

    # Create record
    make_record(dir_path, dist_info_dir)

    # Create wheel
    shutil.make_archive(wheel_name, 'zip', dir_path)
    shutil.move(wheel_name + '.zip', wheel_name)
    shutil.rmtree(dir_path)


def build(arc_root, out_root, tail_args):
    os.chdir(os.path.join(arc_root, 'catboost', 'python-package', 'catboost'))

    py_trait = PythonTrait(arc_root, out_root, tail_args)
    ver = get_version(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.py'))
    pkg_name = os.environ.get('CATBOOST_PACKAGE_NAME', 'catboost')

    try_to_build_gpu_version = True
    if '-DHAVE_CUDA=no' in tail_args:
        try_to_build_gpu_version = False

    for task_type in (['GPU', 'CPU'] if try_to_build_gpu_version else ['CPU']):
        try:
            print('Trying to build {} version'.format(task_type), file=sys.stderr)
            cmd = py_trait.gen_cmd() + (['-DHAVE_CUDA=yes'] if task_type == 'GPU' else ['-DHAVE_CUDA=no'])
            print(' '.join(cmd), file=sys.stderr)
            subprocess.check_call(cmd)
            print('Build {} version: OK'.format(task_type), file=sys.stderr)
            src = os.path.join(py_trait.out_root, 'catboost', 'python-package', 'catboost', py_trait.so_name())
            dst = '.'.join([src, task_type])
            shutil.move(src, dst)
            wheel_name = os.path.join(py_trait.arc_root, 'catboost', 'python-package', '{}-{}-{}-none-{}.whl'.format(pkg_name, ver, py_trait.lang, py_trait.platform))
            make_wheel(wheel_name, pkg_name, ver, arc_root, dst)
            os.remove(dst)
            return wheel_name
        except Exception as e:
            print('{} version build failed: {}'.format(task_type, e), file=sys.stderr)
    raise Exception('Nothing built')


if __name__ == '__main__':
    arc_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    out_root = tempfile.mkdtemp()
    wheel_name = build(arc_root, out_root, sys.argv[1:])
    print(wheel_name)
