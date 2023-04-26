#!/usr/bin/env python3
#
# Build dynamic library with JNI using user-provided arguments and place it to resources directory
# of Maven package
#
# How to use:
#   build_native_for_maven.py
#       --build-system=<build_system>
#       --build-output-root-dir=<build-output-root-dir>
#       --base-dir=<Maven package basedir>
#       --lib-name=<library_name>
#       [<extra ya make or CMake argments...>]
#


from __future__ import absolute_import, print_function

import argparse
import errno
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile


def _get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    return sys.platform

def _get_arch():
    machine = platform.machine()
    if machine.lower() == 'amd64':
        return 'x86_64'
    return machine

def _re_match_line_in_file(file_name: str, regexp: str):
    matcher = re.compile(regexp)
    with open(file_name) as f:
        for line in f.readlines():
            if matcher.match(line[:-1]) is not None:
                return True
    return False

def _get_arcadia_root():
    arcadia_root = None
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    while True:
        cmake_lists_file = os.path.join(path, 'CMakeLists.txt')
        if os.path.isfile(cmake_lists_file):
            if _re_match_line_in_file(cmake_lists_file, "^(project|PROJECT)\("):
                arcadia_root = path
                break

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


def _build_native_platform_to_jvm_platform(build_native_platform):
    target_system, target_arch = build_native_platform.split('-')
    if target_system == 'windows':
        target_system = 'win32'
    if target_arch == 'amd64':
        target_arch = 'x86_64'
    return f'{target_system}-{target_arch}'


def _get_platform_resources_dir(parsed_args):
    if parsed_args.macos_universal_binaries:
        return 'darwin-universal2'
    elif parsed_args.target_platform is not None:
        return _build_native_platform_to_jvm_platform(parsed_args.target_platform)
    else:
        return ''.join((_get_platform(), '-', _get_arch()))

def _is_dll_java(project_path):
    return _re_match_line_in_file(os.path.join(project_path, 'ya.make'), '\W*DLL_JAVA\(.*\)$')


def _extract_classes_from_jar(jar_file, dst_dir):
    with zipfile.ZipFile(jar_file, 'r') as zf:
        for member_name in zf.namelist():
            if member_name.endswith('.class'):
                zf.extract(member_name, dst_dir)

def _get_ya_make_platform_from_build_native_platform(target_platform):
    platform, arch = target_platform.lower().split('-')
    if arch == 'amd64':
        arch = 'x86_64'
    if platform == 'windows':
        # TODO(akhropov): legacy logic, switch to 'win'
        platform = 'win'
    return 'default-' + platform + '-' + arch


def build_shared_lib_with_ya(parsed_args, top_src_root_dir, package_src_sub_path, extra_ya_make_args):
    print('building dynamic library with `ya`', file=sys.stderr)
    sys.stderr.flush()

    native_lib_dir = _get_native_lib_dir(top_src_root_dir, package_src_sub_path)

    env = os.environ.copy()

    ya_path = _get_ya_path()
    ya_make_commands = []

    common_ya_make_args = ([sys.executable, ya_path, 'make', native_lib_dir]
        + ['--output', parsed_args.build_output_root_dir]
        + ['--build=' + parsed_args.build_type.lower()]
        + ['-D', 'CATBOOST_OPENSOURCE=yes']
        + ['-D', 'CFLAGS=-DCATBOOST_OPENSOURCE=yes']
        + ['--no-src-links']
        + ['-DOS_SDK=local']
        + ['-DUSE_LOCAL_SWIG=yes']
        + [f'-DUSE_SYSTEM_JDK={os.environ["JAVA_HOME"]}']
        + [f'-DJAVA_HOME={os.environ["JAVA_HOME"]}']
        + [f'-DHAVE_CUDA={"yes" if parsed_args.have_cuda else "no"}']
    )

    cpp_ya_make_extra_args = []
    if parsed_args.macos_universal_binaries:
        cpp_ya_make_extra_args = [
            '--lipo',
            '--target-platform=default-darwin-x86_64',
            '--target-platform=default-darwin-arm64'
        ]
    elif parsed_args.target_platform is not None:
        cpp_ya_make_extra_args = [
            '--target-platform',
             _get_ya_make_platform_from_build_native_platform(parsed_args.target_platform)
        ]

    if parsed_args.macos_universal_binaries and _is_dll_java(native_lib_dir):
        # TODO(akhropov): run separate commands until DLL_JAVA is fully compatible with '--lipo' (YA-60).

        ya_make_commands.append(
            common_ya_make_args + cpp_ya_make_extra_args + extra_ya_make_args + ['-D', 'BUILD_LANGUAGES=CPP']
        )
        ya_make_commands.append(
            common_ya_make_args
            + extra_ya_make_args
            + ['-D', 'BUILD_LANGUAGES=JAVA']
        )
    else:
        ya_make_commands.append(common_ya_make_args + cpp_ya_make_extra_args + extra_ya_make_args)

    for ya_make_command in ya_make_commands:
        if parsed_args.verbose:
            print (' '.join(ya_make_command))
        subprocess.check_call(
            ya_make_command,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr)

def postprocess_after_ya(native_lib_build_dir, lib_name, resources_dir, base_dir):
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

        os.makedirs(target_dir, exist_ok=True)

        shutil.copy(os.path.join(native_lib_build_dir, lib_name + '-sources.jar'), target_dir)

def postprocess_after_cmake(parsed_args, top_src_root_dir, resources_dir):
    package_src_sub_path = os.path.relpath(parsed_args.base_dir, top_src_root_dir)
    if parsed_args.macos_universal_binaries:
        # build_native.build does not produce anything except final binaries in standard native_lib_build_dir place
        # so get java sources from one of the architecture subdirs
        native_lib_build_dir = _get_native_lib_dir(
            os.path.join(parsed_args.build_output_root_dir, 'darwin-x86_64'),
            package_src_sub_path
        )
    else:
        native_lib_build_dir = _get_native_lib_dir(parsed_args.build_output_root_dir, package_src_sub_path)

    file_with_swig_generated_sources = os.path.join(native_lib_build_dir, "swig_gen_java.lst")
    if os.path.exists(file_with_swig_generated_sources):
        print('compile generated java code to resources dir', file=sys.stderr)
        subprocess.check_call(
            ['javac', '-d',  resources_dir, f'@{file_with_swig_generated_sources}']
        )

        target_dir = os.path.join(parsed_args.base_dir, 'target')
        os.makedirs(target_dir, exist_ok=True)

        # sources jar is needed for documentation generators
        print('create sources jar in target', file=sys.stderr)
        subprocess.check_call(
            [
                'jar',
                'cMf', os.path.join(target_dir, parsed_args.lib_name + '-sources.jar'),
                '-C', os.path.join(native_lib_build_dir, 'java'),
                'ru'
            ]
        )


def _main():
    args_parser = argparse.ArgumentParser(allow_abbrev=False)
    args_parser.add_argument('--build-system', choices=['CMAKE', 'YA'], default='CMAKE')
    args_parser.add_argument(
        '--only-postprocessing',
        help="Don't re-run native build if artifacts have already been built in build-output-root-dir",
        action='store_true'
    )
    args_parser.add_argument('--base-dir', required=True, help="Maven package base dir (that contains pom.xml)")
    args_parser.add_argument('--lib-name', required=True)
    args_parser.add_argument('--build-output-root-dir', help="Root of the build output dir", required=True)
    args_parser.add_argument('--build-type', help='build type (Debug,Release,RelWithDebInfo,MinSizeRel)', default='Release')
    args_parser.add_argument('--verbose', help='Verbose output', action='store_true')
    args_parser.add_argument('--have-cuda', help='Enable CUDA support', action='store_true')
    args_parser.add_argument('--cuda-root-dir', help='CUDA root dir (taken from CUDA_PATH or CUDA_ROOT by default)')
    args_parser.add_argument('--macos-universal-binaries', help='Build macOS universal binaries', action='store_true')
    args_parser.add_argument('--target-platform', help='Target platform to build for (like "linux-aarch64"), same as host platform by default')
    parsed_args, extra_args = args_parser.parse_known_args()

    top_src_root_dir = _get_arcadia_root()

    package_src_sub_path = os.path.relpath(parsed_args.base_dir, top_src_root_dir)

    resources_dir = _get_package_resources_dir(parsed_args.base_dir)
    os.makedirs(resources_dir, exist_ok=True)
    shared_lib_dir = os.path.join(
        resources_dir,
        _get_platform_resources_dir(parsed_args),
        'lib')
    os.makedirs(shared_lib_dir, exist_ok=True)

    native_lib_build_dir = _get_native_lib_dir(parsed_args.build_output_root_dir, package_src_sub_path)

    if parsed_args.build_system == 'YA':
        if not parsed_args.only_postprocessing:
            build_shared_lib_with_ya(
                parsed_args,
                top_src_root_dir,
                package_src_sub_path,
                extra_args
            )
        postprocess_after_ya(native_lib_build_dir, parsed_args.lib_name, resources_dir, parsed_args.base_dir)
    else: # CMAKE
        if not parsed_args.only_postprocessing:
            sys.path = [os.path.join(top_src_root_dir, 'build')] + sys.path

            import build_native

            build_native.build(
                build_root_dir=parsed_args.build_output_root_dir,
                targets=[parsed_args.lib_name],
                build_type=parsed_args.build_type,
                verbose=parsed_args.verbose,
                have_cuda=parsed_args.have_cuda,
                cuda_root_dir=parsed_args.cuda_root_dir,
                target_platform=parsed_args.target_platform,
                macos_universal_binaries=parsed_args.macos_universal_binaries,
                cmake_extra_args=extra_args
            )

        postprocess_after_cmake(parsed_args, top_src_root_dir, resources_dir)

    native_lib_name = {
        'darwin': 'lib{}.dylib',
        'win32': '{}.dll',
        'linux': 'lib{}.so',
    }[_get_platform()].format(parsed_args.lib_name)

    print('copying dynamic library to resources/lib', file=sys.stderr)
    shutil.copy(
        os.path.join(native_lib_build_dir, native_lib_name),
        shared_lib_dir)

if '__main__' == __name__:
    _main()
