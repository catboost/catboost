import distutils
import logging
import os
import shutil
import subprocess
import sys
from distutils.command.bdist import bdist as _bdist
from distutils.command.build import build as _build
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

SETUP_DIR = os.path.abspath(os.path.dirname(__file__))
PKG_INFO = 'PKG-INFO'
EXT_SRC = 'catboost_so_src'


def get_topsrc_dir():
    if os.path.exists(os.path.join(SETUP_DIR, PKG_INFO)):
        return os.path.join(SETUP_DIR, EXT_SRC)
    else:
        return os.path.abspath(os.path.join(SETUP_DIR, '..', '..'))

def create_hnsw_submodule(verbose, dry_run):
    logging.info('Creating hnsw submodule')
    hnsw_submodule_dir = os.path.join(SETUP_DIR, 'catboost', 'hnsw')
    if not os.path.exists(hnsw_submodule_dir):
        hnsw_original_dir = os.path.join(get_topsrc_dir(), 'library', 'python', 'hnsw', 'hnsw')
        if verbose:
            logging.info(f'create symlink from {hnsw_original_dir} to {hnsw_submodule_dir}')
        if not dry_run:
            # there can be issues on Windows when creating symbolic and hard links
            try:
                os.symlink(hnsw_original_dir, hnsw_submodule_dir, target_is_directory=True)
                return
            except Exception as exception:
                logging.error(f'Encountered an error ({str(exception)}) when creating symlink, try to create hardlink instead')

            if verbose:
                logging.info(f'create hardlink from {hnsw_original_dir} to {hnsw_submodule_dir}')
            try:
                os.link(hnsw_original_dir, hnsw_submodule_dir)
                return
            except Exception as exception:
                logging.error(f'Encountered an error ({str(exception)}) when creating hardlink, just copy instead')

            if verbose:
                logging.info(f'copy from {hnsw_original_dir} to {hnsw_submodule_dir}')
            shutil.copytree(hnsw_original_dir, hnsw_submodule_dir, dirs_exist_ok=True)


def get_all_cmake_lists(base_dir, add_android=True):
    files_list = [
        'CMakeLists.darwin-arm64.txt',
        'CMakeLists.darwin-x86_64.txt',
        'CMakeLists.linux-aarch64.txt',
        'CMakeLists.linux-x86_64-cuda.txt',
        'CMakeLists.linux-x86_64.txt',
        'CMakeLists.windows-x86_64-cuda.txt',
        'CMakeLists.windows-x86_64.txt',
        'CMakeLists.txt',
    ]
    if add_android:
        files_list += [
            'CMakeLists.android-arm.txt',
            'CMakeLists.android-arm64.txt',
            'CMakeLists.android-x86.txt',
            'CMakeLists.android-x86_64.txt',
        ]
    return [os.path.join(base_dir, f) for f in files_list]

def copy_catboost_sources(topdir, pkgdir, verbose, dry_run):
    topnames = [
        'AUTHORS', 'LICENSE', 'CONTRIBUTING.md', 'README.md', 'RELEASE.md',
        'conanfile.txt',
        'build',
        'catboost/cuda',
        'catboost/idl',
        'catboost/libs',
        'catboost/private',
        'catboost/python-package/catboost',
        'cmake',
        'contrib',
        'library',
        'tools',
        'util',
    ]
    topnames += get_all_cmake_lists('')
    topnames += get_all_cmake_lists('catboost')
    topnames += get_all_cmake_lists('catboost/python-package', add_android=False)
    for name in topnames:
        src = os.path.join(topdir, name)
        dst = os.path.join(pkgdir, name)
        if os.path.isdir(src):
            distutils.dir_util.copy_tree(src, dst, verbose=verbose, dry_run=dry_run)
        else:
            distutils.dir_util.mkpath(os.path.dirname(dst))
            distutils.file_util.copy_file(src, dst, update=1, verbose=verbose, dry_run=dry_run)


def emph(s):
    return '\x1b[32m{}\x1b[0m'.format(s)


def guess_catboost_version():
    version_py = os.path.join('catboost', 'version.py')
    exec(compile(open(version_py).read(), version_py, 'exec'))
    return locals()['VERSION']


def logging_execute(cmd, verbose, dry_run):
    if verbose:
        logging.info('EXECUTE: {}'.format(subprocess.list2cmdline(cmd)))
    if not dry_run:
        subprocess.check_call(cmd, universal_newlines=True)


def python_version():
    return sys.version.split()[0]


class Helper(object):
    options = [
        ('with-cuda=', None, emph('Build with CUDA support (cuda-root|no)')),
        ('with-hnsw', None, emph('Build with hnsw as catboost submodule')),
        ('parallel=', 'j', emph('Number of parallel build jobs')),
        ('prebuilt-extensions-build-root-dir=', None, emph('Use extensions from CatBoost project prebuilt with CMake')),
    ]

    def initialize_options(self):
        self.with_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_ROOT') or None
        self.with_hnsw = False
        self.parallel = None
        self.prebuilt_extensions_build_root_dir = None

    def finalize_options(self):
        if os.path.exists(str(self.with_cuda)):
            logging.info("Targeting for CUDA support with {}".format(self.with_cuda))
        else:
            self.with_cuda = None

    def propagate_options(self, origin, subcommand):
        self.distribution.get_command_obj(subcommand).set_undefined_options(
            origin,
            ("with_cuda", "with_cuda"),
            ("with_hnsw", "with_hnsw"),
            ("parallel", "parallel"),
            ('prebuilt_extensions_build_root_dir', 'prebuilt_extensions_build_root_dir')
        )


class build(_build):

    user_options = _build.user_options + Helper.options

    def initialize_options(self):
        _build.initialize_options(self)
        Helper.initialize_options(self)

    def finalize_options(self):
        _build.finalize_options(self)
        Helper.finalize_options(self)

    def run(self):
        Helper.propagate_options(self, "build", "build_ext")
        _build.run(self)


class bdist(_bdist):

    user_options = _bdist.user_options + Helper.options

    def initialize_options(self):
        _bdist.initialize_options(self)
        Helper.initialize_options(self)

    def finalize_options(self):
        _bdist.finalize_options(self)
        Helper.finalize_options(self)

    def run(self):
        Helper.propagate_options(self, "bdist", "build_ext")
        _bdist.run(self)


class bdist_wheel(_bdist_wheel):

    user_options = _bdist_wheel.user_options + Helper.options

    def initialize_options(self):
        _bdist_wheel.initialize_options(self)
        Helper.initialize_options(self)

    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        Helper.finalize_options(self)

    def run(self):
        Helper.propagate_options(self, "bdist_wheel", "build_ext")
        _bdist_wheel.run(self)


class ExtensionWithSrcAndDstSubPath(Extension):
    def __init__(self, name, cmake_build_sub_path, dst_sub_path):
        super().__init__(name, sources=[])
        self.cmake_build_sub_path = cmake_build_sub_path
        self.dst_sub_path = dst_sub_path


class build_ext(_build_ext):

    user_options = _build_ext.user_options + Helper.options

    def initialize_options(self):
        _build_ext.initialize_options(self)
        Helper.initialize_options(self)

    def finalize_options(self):
        _build_ext.finalize_options(self)
        Helper.finalize_options(self)

    @staticmethod
    def get_cmake_built_extension_filename(ext_name):
        return {
            'linux': f'lib{ext_name}.so',
            'darwin': f'lib{ext_name}.dylib',
            'win32': f'{ext_name}.dll',
        }[sys.platform]

    @staticmethod
    def get_extension_suffix():
        return {
            'linux': '.so',
            'darwin': '.so',
            'win32': '.pyd',
        }[sys.platform]

    def run(self):
        verbose = self.distribution.verbose
        dry_run = self.distribution.dry_run

        for ext in self.extensions:
            if not isinstance(ext, ExtensionWithSrcAndDstSubPath):
                raise RuntimeError('Only ExtensionWithSrcAndDstSubPath extensions are supported')

            put_dir = os.path.abspath(os.path.join(self.build_lib, ext.dst_sub_path))
            distutils.dir_util.mkpath(put_dir, verbose=verbose, dry_run=dry_run)

        if self.prebuilt_extensions_build_root_dir is not None:
            build_dir = self.prebuilt_extensions_build_root_dir
        else:
            build_dir = os.path.abspath(self.build_temp)
            self.build_with_cmake_and_ninja(get_topsrc_dir(), build_dir, verbose, dry_run)

        self.copy_artifacts_built_with_cmake(build_dir, verbose, dry_run)

    def build_with_cmake_and_ninja(self, topsrc_dir, build_dir, verbose, dry_run):
        targets = [ext.name for ext in self.extensions]

        logging.info(f'Buildling {",".join(targets)} with cmake and ninja')

        sys.path = [os.path.join(topsrc_dir, 'build')] + sys.path
        import build_native

        python3_root_dir = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir))
        if self.with_cuda:
            cuda_support_msg = 'with CUDA support'
        else:
            cuda_support_msg = 'without CUDA support'

        build_native.build(
            build_root_dir=build_dir,
            targets=targets,
            verbose=verbose,
            dry_run=dry_run,
            have_cuda=bool(self.with_cuda),
            cuda_root_dir=self.with_cuda,
            cmake_extra_args=[f'-DPython3_ROOT_DIR={python3_root_dir}']
        )

        logging.info(f'Successfully built {",".join(targets)} {cuda_support_msg}')

    def copy_artifacts_built_with_cmake(self, build_dir, verbose, dry_run):
        for ext in self.extensions:
            # TODO(akhropov): CMake produces wrong artifact names right now so we have to rename it
            dll = os.path.join(
                build_dir,
                ext.cmake_build_sub_path,
                build_ext.get_cmake_built_extension_filename(ext.name)
            )
            distutils.file_util.copy_file(
                dll,
                os.path.join(self.build_lib, ext.dst_sub_path, ext.name + build_ext.get_extension_suffix()),
                verbose=verbose,
                dry_run=dry_run
            )

class sdist(_sdist):

    def make_release_tree(self, base_dir, files):
        _sdist.make_release_tree(self, base_dir, files)
        copy_catboost_sources(
            os.path.join(SETUP_DIR, '..', '..'),
            os.path.join(base_dir, EXT_SRC),
            verbose=self.distribution.verbose,
            dry_run=self.distribution.dry_run,
        )


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)1.1s %(msg)s', level=logging.DEBUG, datefmt=emph('%X'))

    extensions = [
        ExtensionWithSrcAndDstSubPath(
            '_catboost',
            os.path.join('catboost', 'python-package', 'catboost'),
            'catboost'
        )
    ]
    if '--with-hnsw' in sys.argv:
        create_hnsw_submodule('--verbose' in sys.argv, '--dry-run' in sys.argv)
        extensions.append(
            ExtensionWithSrcAndDstSubPath(
                '_hnsw',
                os.path.join('library', 'python', 'hnsw', 'hnsw'),
                os.path.join('catboost', 'hnsw')
            )
        )

    setup(
        name=os.environ.get('CATBOOST_PACKAGE_NAME') or 'catboost',
        version=os.environ.get('CATBOOST_PACKAGE_VERSION') or guess_catboost_version(),
        packages=find_packages(),
        package_data={
            'catboost.widget': [
                '*.css', '*.js', '*/*.ipynb',
                '*/*/*.json', '*/*/*/*.json',
                '.eslintrc',
            ],
        },
        ext_modules=extensions,
        cmdclass={
            'bdist_wheel': bdist_wheel,
            'bdist': bdist,
            'build_ext': build_ext,
            'build': build,
            'sdist': sdist,
        },
        author='CatBoost Developers',
        author_email='catboost@yandex-team.ru',
        description='CatBoost Python Package',
        long_description='CatBoost is a fast, scalable, high performance gradient boosting on decision trees library. Used for ranking, classification, regression and other ML tasks.',
        license='Apache License, Version 2.0',
        url='https://catboost.ai',
        project_urls={
            'GitHub': 'https://github.com/catboost/catboost',
            'Bug Tracker': 'https://github.com/catboost/catboost/issues',
            'Documentation': 'https://catboost.ai/docs/concepts/about.html',
            'Benchmarks': 'https://catboost.ai/#benchmark',
        },
        keywords=['catboost'],
        platforms=['Linux', 'Mac OSX', 'Windows', 'Unix'],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "License :: OSI Approved :: Apache License 2.0",
        ],
        install_requires=[
            'graphviz',
            'matplotlib',
            'numpy (>=1.16.0)',
            'pandas (>=0.24)',
            'scipy',
            'plotly',
            'six',
        ],
        zip_safe=False,
        setup_requires=['wheel'],
    )
