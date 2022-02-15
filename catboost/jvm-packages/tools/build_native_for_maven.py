#!/usr/bin/env python
#
# Build dynamic library with JNI using user-provided arguments and place it to resources directory
# of Maven package
#
# NOTE: this script must be python2/3 compatible
#
# How to use: build_native_for_maven.py <Maven package basedir> <library_name> [<ya make argments...>]
#


from __future__ import absolute_import, print_function

import contextlib
import errno
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


@contextlib.contextmanager
def _tempdir(prefix=None):
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    yield tmp_dir
    # TODO(yazevnul): log error
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    return sys.platform

def _get_arch():
    machine = platform.machine()
    if machine.lower() == 'amd64':
        return 'x86_64'
    return machine


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


def _get_ya_path():
    ya_path = os.path.join(_get_arcadia_root(), 'ya')
    assert os.path.isfile(ya_path), 'no `ya` in arcadia root'
    assert os.access(ya_path, os.X_OK), '`ya` must be executable'
    return ya_path


def _get_package_resources_dir(base_dir):
    return os.path.join(base_dir, 'src', 'main', 'resources')


def _get_native_lib_dir(root_dir, package_arcadia_path):
    return os.path.join(root_dir, package_arcadia_path, 'src', 'native_impl')


def _ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise


def _transform_yamake_platform(target_platform):
    parts = target_platform.lower().split('-')
    platform = parts[-2]
    arch = parts[-1]
    if arch == 'amd64':
        arch = 'x86_64'
    if platform == 'win':
        # TODO(akhropov): legacy logic, switch to 'win'
        platform = 'win32'
    return platform + '-' + arch


def _find_target_platforms(yamake_args):
    target_platforms = []

    arg_idx = 0
    while arg_idx < len(yamake_args):
        arg = yamake_args[arg_idx]
        if arg.startswith('--target-platform'):
            if len(arg) == len('--target-platform'):
                target_platforms.append(yamake_args[arg_idx + 1])
                arg_idx += 2
            elif arg[len('--target-platform')] == '=':
                target_platforms.append(arg[(len('--target-platform') + 1):])
                arg_idx += 1
        else:
            arg_idx += 1

    return [_transform_yamake_platform(platform) for platform in target_platforms]


def _get_platform_resources_dir(yamake_args):
    target_platforms = _find_target_platforms(yamake_args)
    if not target_platforms:
        return ''.join((_get_platform(), '-', _get_arch()))
    if all([platform.startswith('darwin') for platform in target_platforms]) and ('--lipo' in yamake_args):
        return 'darwin-universal2'
    if len(target_platforms) > 1:
        raise Exception("Can't build for multiple target platforms without `--lipo`")
    return target_platforms[0]


def _is_dll_java(project_path):
    with open(os.path.join(project_path, 'ya.make')) as ya_make_file:
        for line in ya_make_file.readlines():
            if re.match('\W*DLL_JAVA\(.*\)$', line[:-1]) is not None:
                return True
    return False

def _get_single_target_ya_make_args(yamake_args):
    other_args = []
    single_target_args = []

    arg_idx = 0
    while arg_idx < len(yamake_args):
        arg = yamake_args[arg_idx]
        if arg.startswith('--target-platform'):
            if len(arg) == len('--target-platform'):
                if not single_target_args:
                    single_target_args = ['--target-platform', yamake_args[arg_idx + 1]]
                arg_idx += 2
            elif arg[len('--target-platform')] == '=':
                if not single_target_args:
                    single_target_args = ['--target-platform', arg[(len('--target-platform') + 1):]]
                arg_idx += 1
        else:
            if arg != '--lipo':
                other_args.append(arg)
            arg_idx += 1

    return other_args + single_target_args


def _extract_classes_from_jar(jar_file, dst_dir):
    with zipfile.ZipFile(jar_file, 'r') as zf:
        for member_name in zf.namelist():
            if member_name.endswith('.class'):
                zf.extract(member_name, dst_dir)


def _main():
    if len(sys.argv) < 3:
        raise Exception('Required basedir and library_name arguments is not specified')

    base_dir = sys.argv[1]
    lib_name = sys.argv[2]
    package_name = os.path.basename(os.path.abspath(base_dir))
    package_arcadia_path = os.path.relpath(base_dir, _get_arcadia_root())
    ya_path = _get_ya_path()
    resources_dir = _get_package_resources_dir(base_dir)
    _ensure_dir_exists(resources_dir)
    shared_lib_dir = os.path.join(
        resources_dir,
        _get_platform_resources_dir(sys.argv[3:]),
        'lib')
    _ensure_dir_exists(shared_lib_dir)
    native_lib_dir = _get_native_lib_dir(_get_arcadia_root(), package_arcadia_path)
    env = os.environ.copy()

    print('building dynamic library with `ya`', file=sys.stderr)
    sys.stderr.flush()

    with _tempdir(prefix='catboost_build-') as build_output_dir:
        ya_make_commands = []

        common_ya_make_args = ([sys.executable, ya_path, 'make', native_lib_dir]
            + ['--output', build_output_dir]
            + ['-D', 'CATBOOST_OPENSOURCE=yes']
            + ['-D', 'CFLAGS=-DCATBOOST_OPENSOURCE=yes'])

        extra_ya_make_args = sys.argv[3:]
        if ('--lipo' in extra_ya_make_args and _is_dll_java(native_lib_dir)):
            # TODO(akhropov): run separate commands until DLL_JAVA is fully compatible with '--lipo' (YA-60).

            ya_make_commands.append(common_ya_make_args + extra_ya_make_args + ['-D', 'BUILD_LANGUAGES=CPP'])
            ya_make_commands.append(
                common_ya_make_args
                + _get_single_target_ya_make_args(extra_ya_make_args)
                + ['-D', 'BUILD_LANGUAGES=JAVA']
            )
        else:
            ya_make_commands.append(common_ya_make_args + extra_ya_make_args)

        for ya_make_command in ya_make_commands:
            print (' '.join(ya_make_command))
            subprocess.check_call(
                ya_make_command,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr)

        native_lib_build_dir = _get_native_lib_dir(build_output_dir, package_arcadia_path)
        jar_name = lib_name + '.jar'
        jar_src_path = os.path.join(native_lib_build_dir, jar_name)
        if os.path.exists(jar_src_path):
            """
                Ya Make's DLL_JAVA packs classes generated by SWIG into it's own jar,
                put these classes into resource dir to be added in main package's jar.
            """

            print('extract classes from jar to resources', file=sys.stderr)
            _extract_classes_from_jar(jar_src_path, resources_dir)

            """
                Copy jar with sources to target dir (needed for documentation generators)
            """
            print('copy sources jar to target', file=sys.stderr)

            target_dir = os.path.join(base_dir, 'target')
            """
                ensure that target directory exists, can't use exist_ok flag because it is unavailable in
                python 2.7
            """
            try:
                os.makedirs(target_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            shutil.copy(os.path.join(native_lib_build_dir, lib_name + '-sources.jar'), target_dir)

        native_lib_name = {
            'darwin': 'lib{}.dylib',
            'win32': '{}.dll',
            'linux': 'lib{}.so',
        }[_get_platform()].format(lib_name)

        print('copying dynamic library to resources/lib', file=sys.stderr)
        shutil.copy(
            os.path.join(_get_native_lib_dir(build_output_dir, package_arcadia_path), native_lib_name),
            shared_lib_dir)


if '__main__' == __name__:
    _main()
