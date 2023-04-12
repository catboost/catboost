from __future__ import print_function

import glob
import subprocess
import shutil
import os
import stat
import sys
import platform
import tempfile
import hashlib

from base64 import urlsafe_b64encode
from argparse import ArgumentParser
from enum import Enum

sys.dont_write_bytecode = True

PL_LINUX = ['manylinux1_x86_64']
PL_MACOS_X86_64 = [
    'macosx_10_6_intel',
    'macosx_10_9_intel',
    'macosx_10_9_x86_64',
    'macosx_10_10_intel',
    'macosx_10_10_x86_64'
]
PL_MACOS_ARM64 = ['macosx_11_0_arm64', 'macosx_12_0_arm64']
PL_MACOS_UNIVERSAL = ['macosx_10_6_universal2']
PL_WIN = ['win_amd64']

class BUILD_SYSTEM(Enum):
    CMAKE = 1
    YA = 2


class PythonVersion(object):
    def __init__(self, major, minor, from_sandbox=False):
        self.major = major
        self.minor = minor
        self.from_sandbox = from_sandbox


class PythonTrait(object):
    def __init__(self, build_system, arc_root, out_root, tail_args):
        self.build_system = build_system
        self.arc_root = arc_root
        self.out_root = out_root
        self.tail_args = tail_args
        self.python_version = mine_system_python_ver(self.tail_args)
        self.platform_tag_string = mine_platform_tag_string(self.tail_args)
        self.py_config, self.lang = self.get_python_info()

    def gen_cmd(self, arc_path, module_name, task_type):
        if self.build_system == BUILD_SYSTEM.CMAKE:
            cmd = ([
                'cd', arc_root,
                'cmake',
                '-B', self.out_root,
                '-G', '"Unix Makefiles"',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_TOOLCHAIN_FILE=' + os.path.join(self.arc_root, 'catboost', 'build', 'toolchains', 'clang.toolchain'),
                '-DCMAKE_POSITION_INDEPENDENT_CODE=On'
            ] + self.tail_args + [
                '&&'
                'cd', self.tmp_build_dir,
                '&&',
                'make', module_name
            ])
        elif self.build_system == BUILD_SYSTEM.YA:
            cmd = [
                sys.executable, arc_root + '/ya', 'make', os.path.join(arc_root, arc_path),
                '--no-src-links', '-r', '--output', out_root, '-DPYTHON_CONFIG=' + self.py_config, '-DNO_DEBUGINFO', '-DOS_SDK=local',
                '-DHAVE_CUDA=yes' if task_type == 'GPU' else '-DHAVE_CUDA=no'
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

    def built_so_name(self, module_name):
        if self._on_win():
            return module_name + '.pyd'

        if self.build_system == BUILD_SYSTEM.CMAKE:
            return 'lib' + module_name + '.so'
        elif self.build_system == BUILD_SYSTEM.YA:
            return module_name + '.so'

    def dst_so_name(self, module_name):
        if self._on_win():
            return module_name + '.pyd'

        return module_name + '.so'

    def dll_ext(self):
        if self._on_win():
            return '.pyd'
        return '.so'

    def _on_win(self):
        if self.platform_tag_string == PL_WIN[0]:
            return True
        return platform.system() == 'Windows'


def mine_platform_tag_string(tail_args):
    target_platforms = find_target_platforms(tail_args)
    platform_tags = transform_target_platforms(target_platforms) if target_platforms else gen_platform_tags()
    if all([tag.startswith('macos') for tag in platform_tags]) and ('--lipo' in tail_args):
        platform_tags = PL_MACOS_UNIVERSAL
    return '.'.join(sorted(platform_tags))


def gen_platform_tags():
    import distutils.util

    value = distutils.util.get_platform().replace("linux", "manylinux1")
    value = value.replace('-', '_').replace('.', '_')
    if 'macosx' in value:
        return PL_MACOS_X86_64
    return [value]


def find_target_platforms(tail_args):
    target_platforms = []

    arg_idx = 0
    while arg_idx < len(tail_args):
        arg = tail_args[arg_idx]
        if arg.startswith('--target-platform'):
            if len(arg) == len('--target-platform'):
                target_platforms.append(tail_args[arg_idx + 1])
                arg_idx += 2
            elif arg[len('--target-platform')] == '=':
                target_platforms.append(arg[(len('--target-platform') + 1):])
                arg_idx += 1
        else:
            arg_idx += 1

    return [platform.lower() for platform in target_platforms]


def transform_target_platforms(target_platforms):
    platform_tags = set()
    for platform in target_platforms:
        if 'linux' in platform:
            platform_tags = platform_tags.union(PL_LINUX)
        elif 'darwin' in platform:
            if 'arm64' in platform:
               platform_tags = platform_tags.union(PL_MACOS_ARM64)
            else:
               platform_tags = platform_tags.union(PL_MACOS_X86_64)
        elif 'win' in platform:
            platform_tags = platform_tags.union(PL_WIN)
        else:
            raise Exception('Unsupported platform {}'.format(platform))
    return list(platform_tags)

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
                record.write(item[tmp_dir_length:] + ',sha256=' + urlsafe_b64encode(calc_sha256_digest(item)).decode('ascii').rstrip('=') + ',' + str(os.path.getsize(item)) + '\n')
            else:
                record.write(item[tmp_dir_length:] + ',,\n')


def make_wheel(wheel_name, pkg_name, ver, build_system, arc_root, dst_so_modules, should_build_widget):
    with tempfile.TemporaryDirectory() as dir_path:
        # Create py files
        os.makedirs(os.path.join(dir_path, pkg_name))

        catboost_package_dir = os.path.join(arc_root, 'catboost/python-package')
        for file_name in ['__init__.py', 'version.py', 'core.py', 'datasets.py', 'utils.py', 'eval', 'widget/__init__.py',
                          'widget/ipythonwidget.py','widget/metrics_plotter.py', 'widget/callbacks.py',
                          'metrics.py', 'monoforest.py', 'plot_helpers.py', 'text_processing.py']:
            src = os.path.join(catboost_package_dir, 'catboost', file_name)
            dst = os.path.join(dir_path, pkg_name, file_name)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))

            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

        hnsw_package_dir = os.path.join(arc_root, 'library/python/hnsw/hnsw')
        hnsw_dst_dir = os.path.join(dir_path, pkg_name, 'hnsw')
        os.makedirs(hnsw_dst_dir)
        for file_name in ['__init__.py', 'hnsw.py']:
            src = os.path.join(hnsw_package_dir, file_name)
            dst = os.path.join(hnsw_dst_dir, file_name)
            shutil.copy(src, dst)

        # Create so files
        py_trait = PythonTrait(build_system, '', '', [])
        for module_name, (built_so_path, dst_subdir) in dst_so_modules.items():
            dst_so_name = py_trait.dst_so_name(module_name)
            shutil.copy(built_so_path, os.path.join(dir_path, pkg_name, dst_subdir, dst_so_name))

        # Create metadata
        dist_info_dir = os.path.join(dir_path, '{}-{}.dist-info'.format(pkg_name, ver))
        shutil.copytree(os.path.join(catboost_package_dir, 'for_mk_wheel', 'catboost.dist-info'), dist_info_dir)

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

        if should_build_widget:
            data_dir = os.path.join(dir_path, '{}-{}.data'.format(pkg_name, ver), 'data')
            widget_dir = os.path.join(catboost_package_dir, 'catboost', 'widget')
            for file in ['extension.js', 'index.js']:
                src = os.path.join(widget_dir, 'js', 'nbextension', file)
                dst = os.path.join(data_dir, 'share', 'jupyter', 'nbextensions', 'catboost-widget', file)
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))
                shutil.copy(src, dst)

            labextension_dir = os.path.join(catboost_package_dir, 'catboost', 'widget', 'js', 'labextension')
            for file in os.listdir(labextension_dir):
                src = os.path.join(labextension_dir, file)
                dst = os.path.join(data_dir, 'share', 'jupyter', 'labextensions', 'catboost-widget', file)
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)

            src = os.path.join(widget_dir, 'catboost-widget.json')
            dst = os.path.join(data_dir, 'etc', 'jupyter', 'nbconfig', 'notebook.d', 'catboost-widget.json')
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            shutil.copy(src, dst)

        # Create record
        make_record(dir_path, dist_info_dir)

        # Create wheel
        shutil.make_archive(wheel_name, 'zip', dir_path)
        shutil.move(wheel_name + '.zip', wheel_name)


def build_widget(arc_root):
    js_dir = os.path.join(arc_root, 'catboost', 'python-package', 'catboost', 'widget', 'js')
    subprocess.check_call('"{}" -m pip install -U jupyterlab'.format(sys.executable), shell=True, cwd=js_dir)
    subprocess.check_call('yarn clean', shell=True, cwd=js_dir)
    subprocess.check_call('yarn install', shell=True, cwd=js_dir)
    subprocess.check_call('yarn build', shell=True, cwd=js_dir)
    # workaround for https://github.com/yarnpkg/yarn/issues/6685
    for directory in glob.glob(os.path.join(tempfile.gettempdir(), 'yarn--*')):
        shutil.rmtree(directory, ignore_errors=True)


def build(build_system, arc_root, out_root, tail_args, should_build_widget, should_build_with_cuda):
    os.chdir(os.path.join(arc_root, 'catboost', 'python-package', 'catboost'))

    py_trait = PythonTrait(build_system, arc_root, out_root, tail_args)
    ver = get_version(os.path.join(os.getcwd(), 'version.py'))
    pkg_name = os.environ.get('CATBOOST_PACKAGE_NAME', 'catboost')

    for task_type in (['GPU', 'CPU'] if should_build_with_cuda else ['CPU']):
        try:
            print('Trying to build {} version'.format(task_type), file=sys.stderr)

            dst_so_modules = {}
            for module_name, arc_path, dst_subdir in (
                ('_catboost', os.path.join('catboost', 'python-package', 'catboost'), ''),
                ('_hnsw', os.path.join('library', 'python', 'hnsw', 'hnsw'), 'hnsw')
            ):
                print('Trying to build {} native library'.format(module_name), file=sys.stderr)

                cmd = py_trait.gen_cmd(arc_path, module_name, task_type)
                print(' '.join(cmd), file=sys.stderr)
                subprocess.check_call(cmd)
                print('Build {} native library: OK'.format(module_name), file=sys.stderr)
                src = os.path.join(py_trait.out_root, arc_path, py_trait.built_so_name(module_name))
                dst = '.'.join([src, task_type])
                shutil.move(src, dst)
                dst_so_modules[module_name] = (dst, dst_subdir)

            print('Build {} version: OK'.format(task_type), file=sys.stderr)
            if should_build_widget:
                build_widget(arc_root)
            wheel_name = os.path.join(py_trait.arc_root, 'catboost', 'python-package',
                                      '{}-{}-{}-none-{}.whl'.format(pkg_name, ver, py_trait.lang, py_trait.platform_tag_string))
            make_wheel(wheel_name, pkg_name, ver, build_system, arc_root, dst_so_modules, should_build_widget)
            return wheel_name
        except Exception as e:
            print('{} version build failed: {}'.format(task_type, e), file=sys.stderr)
    raise Exception('Nothing built')


if __name__ == '__main__':
    arc_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    with tempfile.TemporaryDirectory() as out_root:
        args_parser = ArgumentParser()
        args_parser.add_argument('--build-with-cuda', choices=['yes', 'no'], default='yes')
        args_parser.add_argument('--build-widget', choices=['yes', 'no'], default='yes')
        args_parser.add_argument('--build-system', choices=['CMAKE', 'YA'], default='CMAKE')

        args, tail_args = args_parser.parse_known_args()
        build_system = BUILD_SYSTEM[args.build_system]

        wheel_name = build(build_system, arc_root, out_root, tail_args, args.build_widget == 'yes', args.build_with_cuda == 'yes')
        print(wheel_name)
