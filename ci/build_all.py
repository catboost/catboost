#!/usr/bin/env python3

# This script has to be run in CI from CatBoost source tree root
#
# For python3.12 it requires 'setuptools' package to be installed (for 'distutils')
#
# Environment variables used:
#
#  - GITHUB_ACTION: if it is defined it means this script is run inside GitHub action.
#    Used to adjust standard paths to dependencies
#  - CMAKE_BUILD_ENV_ROOT (optional):
#    path to the root directory with platform-specific subdirectories with dependencies data for CMake
#  - MAKE_BUILD_CACHE_DIR (optional): Use build artifacts cache if specified
#  - HOME (on Linux and macOS): To derive CMAKE_BUILD_ENV_ROOT path if it has not been specified explicitly
#  - USERPROFILE (on Windows): To derive CMAKE_BUILD_ENV_ROOT path if it has not been specified explicitly
#
# Expects needed components installed in predefined places:
#  - CUDA (on Linux and Windows)
#  - JDK
#  - Python development artifacts for specified PYTHON_VERSIONS
#

import argparse
import distutils
import hashlib
import logging
import os
import platform
import subprocess
import sys
import tarfile
from typing import List, Tuple


IS_IN_GITHUB_ACTION = 'GITHUB_ACTION' in os.environ

PYTHON_VERSIONS = [
    (3,8),
    (3,9),
    (3,10),
    (3,11),
    (3,12)
]

MSVS_VERSION = '2022'
MSVC_TOOLSET = '14.29.30133'


if sys.platform == 'win32':
    CMAKE_BUILD_ENV_ROOT = os.environ.get(
        'CMAKE_BUILD_ENV_ROOT',
        os.path.join(os.environ['USERPROFILE'], 'cmake_build_env_root')
    )

    # without C: because we use CMAKE_FIND_ROOT_PATH for speeding up build for many pythons
    if IS_IN_GITHUB_ACTION:
        CUDA_ROOT = '/CUDA/v11.8'
        JAVA_HOME = '/jdk-8'
    else:
        CUDA_ROOT = '/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8'
        JAVA_HOME = '/Program Files/Eclipse Adoptium/jdk-8.0.362.9-hotspot/'
else:
    CMAKE_BUILD_ENV_ROOT = os.environ.get(
        'CMAKE_BUILD_ENV_ROOT',
        os.path.join(os.environ['HOME'], 'cmake_build_env_root')
    )
    if sys.platform == 'linux':
        CUDA_ROOT = '/usr/local/cuda-11'
        JAVA_HOME = '/opt/jdk/8'
    elif sys.platform == 'darwin':
        JAVA_HOME = '/Library/Java/JavaVirtualMachines/jdk-8/Contents/Home/'
        CUDA_ROOT = None


def need_to_build_with_cuda_for_main_targets(platform_name: str):
    system, _ = platform_name.split('-')
    return system in ['linux', 'windows']


def get_primary_platform_name():
    if sys.platform == 'darwin':
        return 'darwin-universal2'
    else:
        return {
            'win32': 'windows',
            'linux': 'linux'
        }[sys.platform] + '-x86_64'

def get_native_platform_name():
    system_name = 'windows' if sys.platform == 'win32' else sys.platform
    arch = platform.machine()
    if arch == 'AMD64':
        arch = 'x86_64'
    return system_name + '-' + arch


# Unfortunately CMake's FindPython does not work reliably in all cases so we have to recreate similar logic here.

def get_python_root_dir(py_ver: Tuple[int, int])-> str:
    # returns python_root_dir relative to CMAKE_FIND_ROOT_PATH

    if sys.platform == 'win32':
        if IS_IN_GITHUB_ACTION:
            # we've created x.y aliases for convinience
            return os.path.join('Python', f'{py_ver[0]}.{py_ver[1]}')
        else:
            # pyenv installs x.y.z versions but we've created x.y aliases for convinience
            return os.path.join('.pyenv', 'pyenv-win', 'versions', f'{py_ver[0]}.{py_ver[1]}')
    if sys.platform == 'darwin':
        if IS_IN_GITHUB_ACTION:
            # we've created x.y aliases for convinience
            return os.path.join('Python', f'{py_ver[0]}.{py_ver[1]}')
        else:
            # pyenv installs x.y.z versions but we've created x.y aliases for convinience
            return os.path.join('.pyenv', 'versions', f'{py_ver[0]}.{py_ver[1]}')
    if sys.platform == 'linux':
        py_ver_str = f'{py_ver[0]}{py_ver[1]}'
        # manylinux2014 image conventions
        return os.path.join('python', f'cp{py_ver_str}-cp{py_ver_str}')


def get_python_version_include_and_library_paths(py_ver: Tuple[int, int]) -> Tuple[str, str]:
    # returns (python include path, python library path) relative to CMAKE_FIND_ROOT_PATH

    base_path = get_python_root_dir(py_ver)

    if sys.platform == 'win32':
        return (os.path.join(base_path, 'include'), os.path.join(base_path, 'libs', f'python{py_ver[0]}{py_ver[1]}.lib'))

    python_sub_name = f'python{py_ver[0]}.{py_ver[1]}'

    if sys.platform == 'darwin':
        lib_sub_path = os.path.join(
            'lib',
            f'python{py_ver[0]}.{py_ver[1]}',
            f'config-{py_ver[0]}.{py_ver[1]}-darwin'
        )
    elif sys.platform == 'linux':
        lib_sub_path = 'lib'

    return (
        os.path.join(base_path, 'include', python_sub_name),
        os.path.join(base_path, lib_sub_path, f'lib{python_sub_name}.a')
    )

def run_in_python_package_dir(
    src_root_dir:str,
    dry_run:bool,
    verbose:bool,
    commands: List[List[str]]):

    os.chdir(os.path.join(src_root_dir, 'catboost', 'python-package'))

    for cmd in commands:
        if verbose:
            logging.info(' '.join(cmd))
        if not dry_run:
            subprocess.check_call(cmd)

    os.chdir(src_root_dir)

def run_with_native_python_with_version_in_python_package_dir(
    src_root_dir:str,
    dry_run:bool,
    verbose:bool,
    py_ver: Tuple[int, int],
    python_cmds_args: List[List[str]]):

    base_path = os.path.join(CMAKE_BUILD_ENV_ROOT, get_native_platform_name(), get_python_root_dir(py_ver))
    if sys.platform == 'win32':
        python_bin_path = os.path.join(base_path, 'python.exe')
    else:
        python_bin_path = os.path.join(base_path, 'bin', 'python')

    run_in_python_package_dir(
        src_root_dir,
        dry_run,
        verbose,
        [[python_bin_path] + cmd_args for cmd_args in python_cmds_args]
    )


def patch_sources(src_root_dir: str, build_test_tools:bool = False, dry_run:bool = False, verbose:bool = False):
    # TODO(akhropov): Remove when system cuda.cmake is updated for Linux cross-build
    distutils.file_util.copy_file(
        src=os.path.join(src_root_dir, 'ci', 'cmake', 'cuda.cmake'),
        dst=os.path.join(src_root_dir, 'cmake', 'cuda.cmake'),
        verbose=verbose,
        dry_run=dry_run
    )


def get_python_plat_name(platform_name: str):
    system, arch = platform_name.split('-')
    if system == 'windows':
        return 'win_amd64'
    elif system == 'darwin':
        return 'macosx_11_0_universal2'
    else: # linux
        return 'manylinux2014_' + arch


def build_r_package(
    src_root_dir: str,
    build_native_root_dir: str,
    with_cuda: bool,
    platform_name: str,
    dry_run: bool,
    verbose: bool
):
    system, _ = platform_name.split('-')

    def get_catboostr_artifact_src_and_dst_name(system: str):
        return {
            'linux': ('libcatboostr.so', 'libcatboostr.so'),
            'darwin': ('libcatboostr.dylib', 'libcatboostr.so'),
            'windows': ('catboostr.dll', 'libcatboostr.dll')
        }[system]

    os.chdir(os.path.join(src_root_dir, 'catboost', 'R-package'))

    if not dry_run:
        os.makedirs('catboost', exist_ok=True)

    entries = [
        'DESCRIPTION',
        'NAMESPACE',
        'README.md',
        'R',
        'inst',
        'man',
        'tests'
    ]
    for entry in entries:
        if os.path.isdir(entry):
            distutils.dir_util.copy_tree(entry, os.path.join('catboost', entry), verbose=verbose, dry_run=dry_run)
        else:
            distutils.file_util.copy_file(entry, os.path.join('catboost', entry), verbose=verbose, dry_run=dry_run)

    binary_dst_dir = os.path.join('catboost', 'inst', 'libs')
    if system == 'windows':
        binary_dst_dir = os.path.join(binary_dst_dir, 'x64')

    if not dry_run:
        os.makedirs(binary_dst_dir, exist_ok=True)

    src, dst = get_catboostr_artifact_src_and_dst_name(system)
    full_src = os.path.join(
        build_native_root_dir,
        'have_cuda' if with_cuda else 'no_cuda',
        platform_name,
        'catboost',
        'R-package',
        'src',
        src
    )
    full_dst = os.path.join(binary_dst_dir, dst)
    if dry_run:
        logging.info(f'copying {full_src} -> {full_dst}')
    else:
        distutils.file_util.copy_file(full_src, full_dst, verbose=verbose, dry_run=dry_run)

    # some R versions on macOS use 'dylib' extension
    if system == 'darwin':
        full_dst = os.path.join(binary_dst_dir, 'libcatboostr.dylib')
        if dry_run:
            logging.info(f'making a symlink {dst} -> {full_dst}')
        else:
            os.symlink(dst, full_dst)

    r_package_file_name = f'catboost-R-{platform_name}.tgz'
    logging.info(f'creating {r_package_file_name}')
    if not dry_run:
        with tarfile.open(r_package_file_name, "w:gz") as tar:
            tar.add('catboost', arcname=os.path.basename('catboost'))

    os.chdir(src_root_dir)


def build_jvm_artifacts(
    src_root_dir: str,
    build_native_root_dir: str,
    platform_name: str,
    macos_universal_binaries:bool,
    build_with_cuda_for_main_targets: str,
    dry_run: bool,
    verbose: bool):

    os.chdir(src_root_dir)

    parts = [
        (os.path.join('catboost', 'jvm-packages', 'catboost4j-prediction'), 'catboost4j-prediction', build_with_cuda_for_main_targets),
        (os.path.join('catboost', 'spark', 'catboost4j-spark', 'core'), 'catboost4j-spark-impl', False)
    ]

    for base_dir, lib_name, with_cuda in parts:
        cuda_status_prefix = 'have_cuda' if with_cuda else 'no_cuda'
        build_dir = os.path.join(build_native_root_dir, cuda_status_prefix, platform_name)

        cmd = [
            'python3',
            os.path.join('catboost', 'jvm-packages', 'tools', 'build_native_for_maven.py'),
            '--only-postprocessing',
            '--base-dir', base_dir,
            '--lib-name', lib_name,
            '--build-output-root-dir', build_dir,
        ]
        if verbose:
            cmd += ['--verbose']
        if with_cuda:
            cmd += ['--have-cuda', f'--cuda-root-dir="{CUDA_ROOT}"']
        if macos_universal_binaries:
            cmd += ['--macos-universal-binaries']
        else:
            cmd += ['--target-platform', platform_name]

        if verbose:
            logging.info(' '.join(cmd))
        if not dry_run:
            subprocess.check_call(cmd)


def get_exe_files(system:str, name:str) -> List[str]:
    return [name + '.exe' if system == 'windows' else name]

def get_static_lib_files(system:str, name:str) -> List[str]:
    prefix = '' if system == 'windows' else 'lib'
    suffix = '.lib' if system == 'windows' else '.a'
    return [prefix + name + sub_suffix + suffix for sub_suffix in ['', '.global']]

def get_shared_lib_files(system:str, name:str) -> List[str]:
    if system == 'windows':
        return [name + '.lib', name + '.dll']
    else:
        suffix = '.so' if system == 'linux' else '.dylib'
        return ['lib' + name + suffix]

def copy_built_artifacts_to_canonical_place(
    platform_name: str,
    with_cuda:bool,
    build_native_root_dir:str,
    built_output_root_dir:str,
    build_test_tools:bool,
    dry_run:bool,
    verbose: bool
):
    """
    Copy only artifacts that are not copied already by postprocessing in building JVM, R and Python packages
    """
    if build_native_root_dir == built_output_root_dir:
        if verbose:
            print(
                f'copy_built_artifacts_to_canonical_place: build_native_root_dir =='
                + f'built_output_root_dir =\n{build_native_root_dir}\n'
                +  'Do not copy to itself'
            )
            return

    system = platform_name.split('-')[0]
    cuda_status_prefix = 'have_cuda' if with_cuda else 'no_cuda'

    artifacts = [
        (cuda_status_prefix, os.path.join('catboost', 'app'), get_exe_files(system, 'catboost')),
        (cuda_status_prefix, os.path.join('catboost', 'libs', 'model_interface'), get_shared_lib_files(system, 'catboostmodel')),
        (cuda_status_prefix, os.path.join('catboost', 'libs', 'model_interface', 'static'), get_static_lib_files(system, 'catboostmodel_static')),
        (cuda_status_prefix, os.path.join('catboost', 'libs', 'train_interface'), get_shared_lib_files(system, 'catboost')),
    ]

    if build_test_tools:
        artifacts += [
            ('no_cuda', os.path.join('catboost', 'tools', 'limited_precision_dsv_diff'), get_exe_files(system, 'limited_precision_dsv_diff')),
            ('no_cuda', os.path.join('catboost', 'tools', 'limited_precision_json_diff'), get_exe_files(system, 'limited_precision_json_diff')),
            ('no_cuda', os.path.join('catboost', 'tools', 'model_comparator'), get_exe_files(system, 'model_comparator')),
        ]


    for cuda_status_prefix, sub_path, files in artifacts:
        for f in files:
            src = os.path.join(build_native_root_dir, cuda_status_prefix, platform_name, sub_path, f)
            dst = os.path.join(built_output_root_dir, cuda_status_prefix, platform_name, sub_path, f)
            if dry_run:
                logging.info(f'copying {src} -> {dst}')
            else:
                distutils.dir_util.mkpath(os.path.dirname(dst), verbose=verbose, dry_run=dry_run)
                distutils.file_util.copy_file(src, dst, verbose=verbose, dry_run=dry_run)

def get_real_build_root_dir(src_root_dir:str, built_output_root_dir:str):
    if os.environ.get('CMAKE_BUILD_CACHE_DIR'):
        build_native_root_dir = os.path.join(
            os.environ['CMAKE_BUILD_CACHE_DIR'],
            hashlib.md5(os.path.abspath(src_root_dir).encode('utf-8')).hexdigest()[:10]
        )
        os.makedirs(build_native_root_dir, exist_ok=True)
        return build_native_root_dir
    else:
        return built_output_root_dir


def build_all_for_one_platform(
    src_root_dir:str,
    built_output_root_dir:str,  # will contain 'no_cuda/{platform_name}' and 'have_cuda/{platform_name}' subdirs
    platform_name:str,  # either "{system}-{arch}' of 'darwin-universal2'
    native_built_tools_root_dir:str=None,
    cmake_target_toolchain:str=None,
    conan_build_profile:str=None,
    conan_host_profile:str=None,
    cmake_extra_args:List[str]=None,
    build_test_tools:bool=False,
    dry_run:bool=False,
    verbose:bool=False):

    build_native_root_dir = get_real_build_root_dir(src_root_dir, built_output_root_dir)

    for prefix in ['no_cuda', 'have_cuda']:
        build_dir = os.path.join(build_native_root_dir, prefix, platform_name)
        if not dry_run:
            os.makedirs(build_dir, exist_ok=True)

    sys.path = [os.path.join(src_root_dir, 'build')] + sys.path
    import build_native

    # exclude python-dependent targets that will be built for concrete python
    # and SWIG (which is always w/o CUDA) and includes JVM-only 'catboost4j-spark-impl'
    all_catboost_targets_except_python_and_spark=[
        target for target in build_native.Targets.catboost.keys()
        if target not in ['_hnsw', '_catboost', 'catboost4j-spark-impl', 'catboost4j-spark-impl-cpp']
    ]

    build_with_cuda_for_main_targets = need_to_build_with_cuda_for_main_targets(platform_name)

    default_cmake_extra_args = [f'-DJAVA_HOME={JAVA_HOME}']
    if cmake_extra_args is not None:
        default_cmake_extra_args += cmake_extra_args

    cmake_platform_to_root_path = None
    if platform_name == 'darwin-universal2':
        target_platform = None
        macos_universal_binaries = True
        cmake_platform_to_root_path = dict(
             (platform_name, os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name))
              for platform_name in ['darwin-x86_64', 'darwin-arm64']
        )
    else:
        target_platform = platform_name
        macos_universal_binaries = False
        default_cmake_extra_args += [f'-DCMAKE_FIND_ROOT_PATH={os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name)}']

    def call_build_native(
        targets,
        have_cuda,
        macos_universal_binaries=macos_universal_binaries,
        build_root_dir=None,
        native_built_tools_root_dir=None,
        cmake_extra_args=[],
        cmake_platform_to_python_dev_paths=None
    ):
        if build_root_dir is None:
            build_root_dir = os.path.join(
                build_native_root_dir,
                'have_cuda' if have_cuda else 'no_cuda',
                platform_name
            )

        build_native.build(
            dry_run=dry_run,
            verbose=verbose,
            build_root_dir=build_root_dir,
            targets=targets,
            cmake_target_toolchain=cmake_target_toolchain,
            conan_build_profile=conan_build_profile,
            conan_host_profile=conan_host_profile,
            have_cuda=have_cuda,
            cuda_root_dir=CUDA_ROOT if have_cuda else None,
            target_platform=target_platform,
            msvs_version=MSVS_VERSION,
            msvc_toolset=MSVC_TOOLSET,
            macos_universal_binaries=macos_universal_binaries,
            native_built_tools_root_dir=native_built_tools_root_dir,
            cmake_extra_args=default_cmake_extra_args + cmake_extra_args,
            cmake_platform_to_root_path=cmake_platform_to_root_path,
            cmake_platform_to_python_dev_paths=cmake_platform_to_python_dev_paths
        )

    if not native_built_tools_root_dir:
        # build all tools w/o CUDA (will need them for Spark anyway)
        if macos_universal_binaries:
            native_built_tools_root_dir = os.path.join(
                build_native_root_dir,
                'no_cuda',
                platform_name,
                get_native_platform_name()
            )
        else:
            native_built_tools_root_dir = os.path.join(
                build_native_root_dir,
                'no_cuda',
                platform_name
            )
        if not dry_run:
            os.makedirs(native_built_tools_root_dir, exist_ok=True)

        call_build_native(
            targets=build_native.Targets.tools,
            have_cuda=False,
            build_root_dir=native_built_tools_root_dir,
            macos_universal_binaries=False
        )

    targets_wo_cuda = ['catboost4j-spark-impl-cpp']
    if build_test_tools:
        targets_wo_cuda += build_native.Targets.test_tools.keys()


    # build Spark native part and test tools w/o CUDA
    call_build_native(
        targets=targets_wo_cuda,
        have_cuda=False,
        native_built_tools_root_dir=native_built_tools_root_dir
    )

    # build all non python-version specific variants
    call_build_native(
        targets=all_catboost_targets_except_python_and_spark,
        have_cuda=build_with_cuda_for_main_targets,
        native_built_tools_root_dir=native_built_tools_root_dir
    )

    if os.environ.get('CMAKE_BUILD_CACHE_DIR'):
        copy_built_artifacts_to_canonical_place(
            platform_name,
            build_with_cuda_for_main_targets,
            build_native_root_dir,
            built_output_root_dir,
            build_test_tools=build_test_tools,
            dry_run=dry_run,
            verbose=verbose
        )

    build_r_package(
        src_root_dir,
        build_native_root_dir,
        build_with_cuda_for_main_targets,
        platform_name,
        dry_run,
        verbose
    )

    build_jvm_artifacts(
        src_root_dir,
        build_native_root_dir,
        platform_name,
        macos_universal_binaries,
        build_with_cuda_for_main_targets,
        dry_run,
        verbose
    )

    # build python version-specific wheels.
    # Note: assumes build_widget has already been called
    for py_ver in PYTHON_VERSIONS:
        python_include_path, python_library_path = get_python_version_include_and_library_paths(py_ver)
        if macos_universal_binaries:
            cmake_extra_args=[]
            cmake_platform_to_python_dev_paths = dict(
                (
                    platform_name,
                    build_native.PythonDevPaths(
                        os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name, python_include_path),
                        os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name, python_library_path)
                    )
                )
                for platform_name in ['darwin-x86_64', 'darwin-arm64']
            )
        else:
            cmake_extra_args=[
                f'-DPython3_LIBRARY={os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name, python_library_path)}',
                f'-DPython3_INCLUDE_DIR={os.path.join(CMAKE_BUILD_ENV_ROOT, platform_name, python_include_path)}'
            ]
            cmake_platform_to_python_dev_paths=None

        call_build_native(
            targets=['_hnsw', '_catboost'],
            cmake_extra_args=cmake_extra_args,
            have_cuda=build_with_cuda_for_main_targets,
            native_built_tools_root_dir=native_built_tools_root_dir,
            cmake_platform_to_python_dev_paths=cmake_platform_to_python_dev_paths
        )

        build_native_sub_dir = os.path.join(
            build_native_root_dir,
            'have_cuda' if build_with_cuda_for_main_targets else 'no_cuda',
            platform_name
        )

        # don't pass CUDA_ROOT here because it does not matter when prebuilt extension libraries are used
        bdist_wheel_cmd = [
            'setup.py',
            'bdist_wheel',
            '--plat-name', get_python_plat_name(platform_name),
            '--with-hnsw',
            '--prebuilt-widget',
            f'--prebuilt-extensions-build-root-dir={build_native_sub_dir}'
        ]

        run_with_native_python_with_version_in_python_package_dir(
            src_root_dir,
            dry_run,
            verbose,
            py_ver,
            [bdist_wheel_cmd]
        )

def build_all(src_root_dir: str, build_test_tools:bool = False, dry_run:bool = False, verbose:bool = False):
    run_in_python_package_dir(
        src_root_dir,
        dry_run,
        verbose,
        [
            ['python3', 'setup.py', 'build_widget'],
            ['python3', '-m', 'build', '--sdist']
        ]
    )

    build_native_root_dir = os.path.join(src_root_dir, 'build_native_root')

    platform_name = get_primary_platform_name()

    if platform_name.startswith('linux'):
        cmake_target_toolchain=os.path.join(src_root_dir, 'ci', 'toolchains', 'clangs.toolchain')
        conan_build_profile=os.path.join(src_root_dir, 'ci', 'conan', 'profiles', 'build.manylinux2014.x86_64.profile')
        conan_host_profile=os.path.join(src_root_dir, 'ci', 'conan', 'profiles', 'manylinux2014.x86_64.profile')
    else:
        cmake_target_toolchain=None
        conan_build_profile=None
        conan_host_profile=None

    build_all_for_one_platform(
        src_root_dir=src_root_dir,
        built_output_root_dir=build_native_root_dir,
        platform_name=platform_name,
        cmake_target_toolchain=cmake_target_toolchain,
        conan_build_profile=conan_build_profile,
        conan_host_profile=conan_host_profile,
        build_test_tools=build_test_tools,
        dry_run=dry_run,
        verbose=verbose
    )

    if platform_name.startswith('linux'):
        platform_java_home = os.path.join(CMAKE_BUILD_ENV_ROOT, 'linux-aarch64', JAVA_HOME[1:])

        # build for aarch64 as well
        build_all_for_one_platform(
            src_root_dir=src_root_dir,
            built_output_root_dir=build_native_root_dir,
            platform_name='linux-aarch64',
            cmake_target_toolchain=os.path.join(src_root_dir, 'ci', 'toolchains', 'dockcross.manylinux2014_aarch64.clangs.toolchain'),
            conan_build_profile=conan_build_profile,
            conan_host_profile=os.path.join(src_root_dir, 'ci', 'conan', 'profiles', 'dockcross.manylinux2014_aarch64.profile'),
            build_test_tools=build_test_tools,
            dry_run=dry_run,
            verbose=verbose,
            native_built_tools_root_dir=os.path.join(
                get_real_build_root_dir(src_root_dir, build_native_root_dir),
                'no_cuda',
                'linux-x86_64'
            ),

            # for some reason CMake can't find JDK libraries in Adoptium's standard path so we have to specify them explicitly
            cmake_extra_args=[
                f'-DJAVA_AWT_LIBRARY={os.path.join(platform_java_home, "lib/aarch64/libjawt.so")}',
                f'-DJAVA_JVM_LIBRARY={os.path.join(platform_java_home, "jre/lib/aarch64/server/libjvm.so")}'
            ]
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args_parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description='This script has to be run in CI from CatBoost source tree root'
    )
    args_parser.add_argument('--dry-run', action='store_true', help='Only print, not execute commands')
    args_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args_parser.add_argument('--build-test-tools', action='store_true', help='Build tools for tests')
    parsed_args = args_parser.parse_args()

    patch_sources(
        os.path.abspath(os.getcwd()),
        parsed_args.build_test_tools,
        parsed_args.dry_run,
        parsed_args.verbose
    )
    build_all(
        os.path.abspath(os.getcwd()),
        parsed_args.build_test_tools,
        parsed_args.dry_run,
        parsed_args.verbose
    )
