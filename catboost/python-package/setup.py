import distutils
import itertools
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List
from distutils.spawn import spawn
from distutils.command.bdist import bdist as _bdist

# requires setuptools >= 64.0.0
import setuptools.command.build  # for SubCommand
from setuptools.command.build import build as _build

import setuptools
from setuptools import setup, find_packages, Extension

from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

SETUP_DIR = os.path.abspath(os.path.dirname(__file__))
PKG_INFO = 'PKG-INFO'
EXT_SRC = 'catboost_all_src'


def get_topsrc_dir():
    if os.path.exists(os.path.join(SETUP_DIR, PKG_INFO)):
        return os.path.join(SETUP_DIR, EXT_SRC)
    else:
        return os.path.abspath(os.path.join(SETUP_DIR, '..', '..'))


class ExtensionWithSrcAndDstSubPath(Extension):
    def __init__(self, name, cmake_build_sub_path, dst_sub_path):
        super().__init__(name, sources=[])
        self.cmake_build_sub_path = cmake_build_sub_path
        self.dst_sub_path = dst_sub_path

def setup_hnsw_submodule(argv, extensions):
    """
    Does not respect --dry-run because main setup.py commands won't work correctly without this submodule setup
    """

    cmake_build_sub_path = os.path.join('library', 'python', 'hnsw', 'hnsw')
    dst_sub_path = os.path.join('catboost', 'hnsw')

    hnsw_submodule_dir = os.path.join(SETUP_DIR, dst_sub_path)

    verbose = '--verbose' in argv

    if '--with-hnsw' in argv:
        extensions.append(ExtensionWithSrcAndDstSubPath('_hnsw', cmake_build_sub_path, dst_sub_path))

        if not os.path.exists(hnsw_submodule_dir):
            logging.info('Creating hnsw submodule')

            hnsw_original_dir = os.path.join(get_topsrc_dir(), cmake_build_sub_path)

            if verbose:
                logging.info(f'create symlink from {hnsw_original_dir} to {hnsw_submodule_dir}')

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
    elif os.path.exists(hnsw_submodule_dir):
        if verbose:
            logging.info('remove previously used catboost.hnsw submodule')
        if os.path.islink(hnsw_submodule_dir):
            os.remove(hnsw_submodule_dir)
        elif sys.version_info >= (3, 8):
            shutil.rmtree(hnsw_submodule_dir)
        else:
            raise RuntimeError("Cannot correctly remove previously used 'hnsw' submodule because it might be a directory junction")


def get_setup_requires(argv):
    setup_requires = ['wheel']
    if ('build_widget' in argv) or (not ('--no-widget' in argv)):
        setup_requires += ['jupyterlab']
    return setup_requires


def get_all_cmake_lists(topdir, sub_path):
    return [
        os.path.join(sub_path, f) for f in os.listdir(os.path.join(topdir, sub_path))
        if f.startswith('CMakeLists')
    ]

def copy_catboost_sources(topdir, pkgdir, verbose, dry_run):
    topnames = [
        'AUTHORS', 'LICENSE', 'CONTRIBUTING.md', 'README.md', 'RELEASE.md',
        'conanfile.txt',
        'build',
        os.path.join('catboost', 'cuda'),
        os.path.join('catboost', 'idl'),
        os.path.join('catboost', 'libs'),
        os.path.join('catboost', 'private'),
        os.path.join('catboost', 'python-package', 'catboost'),
        'cmake',
        'contrib',
        'library',
        'tools',
        'util',
    ]
    topnames += get_all_cmake_lists(topdir, '')
    topnames += get_all_cmake_lists(topdir, 'catboost')
    topnames += get_all_cmake_lists(topdir, os.path.join('catboost', 'python-package'))
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


class OptionsHelper(object):
    @staticmethod
    def get_user_options(extra_options_classes):
        return list(itertools.chain.from_iterable([cls.options for cls in extra_options_classes]))

    @staticmethod
    def initialize_options(command):
        for extra_options_class in command.__class__.extra_options_classes:
            extra_options_class.initialize_options(command)

    @staticmethod
    def finalize_options(command):
        for extra_options_class in command.__class__.extra_options_classes:
            extra_options_class.finalize_options(command)

    @staticmethod
    def propagate(command, subcommand_name, options):
        sub_cmd = command.reinitialize_command(subcommand_name, reinit_subcommands=True)

        for opt_name in options:
            setattr(sub_cmd, opt_name, getattr(command, opt_name))


class HNSWOptions(object):
    options = [
        ('with-hnsw', None, emph('Build with hnsw as catboost submodule')),
    ]

    @staticmethod
    def initialize_options(command):
        command.with_hnsw = False

    @staticmethod
    def finalize_options(command):
        pass

    @staticmethod
    def get_options_attribute_names():
        return ['with_hnsw']

class WidgetOptions(object):
    options = [
        ('no-widget', None, emph('Disable Jupyter visualization widget support that is enabled by default')),
    ]

    @staticmethod
    def initialize_options(command):
        command.no_widget = False

    @staticmethod
    def finalize_options(command):
        pass

    def get_options_attribute_names():
        return ['no_widget']

class BuildExtOptions(object):
    options = [
        ('with-cuda=', None, emph('Build with CUDA support (cuda-root|no)')),
        ('parallel=', 'j', emph('Number of parallel build jobs')),
        ('prebuilt-extensions-build-root-dir=', None, emph('Use extensions from CatBoost project prebuilt with CMake')),
        ('macos-universal-binaries', None, emph('Build extension libraries as macOS universal binaries'))
    ]

    @staticmethod
    def initialize_options(command):
        command.with_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_ROOT') or None
        command.parallel = None
        command.prebuilt_extensions_build_root_dir = None
        command.macos_universal_binaries=False

    @staticmethod
    def finalize_options(command):
        if os.path.exists(str(command.with_cuda)):
            logging.info("Targeting for CUDA support with {}".format(command.with_cuda))
        else:
            command.with_cuda = None

    def get_options_attribute_names():
        return ['with_cuda', 'parallel', 'prebuilt_extensions_build_root_dir', 'macos_universal_binaries']


class build(_build):
    extra_options_classes = [HNSWOptions, WidgetOptions, BuildExtOptions]

    user_options = _build.user_options + OptionsHelper.get_user_options(extra_options_classes)

    def initialize_options(self):
        _build.initialize_options(self)
        OptionsHelper.initialize_options(self)

    def finalize_options(self):
        _build.finalize_options(self)
        OptionsHelper.finalize_options(self)

    def run(self):
        OptionsHelper.propagate(
            self,
            "build_ext",
            HNSWOptions.get_options_attribute_names() + BuildExtOptions.get_options_attribute_names()
        )
        _build.run(self)

    def no_widget_option_is_not_set(self):
        return not self.no_widget

    sub_commands = [
        ('build_py',      _build.has_pure_modules),
        ('build_ext',     _build.has_ext_modules),
        ('build_scripts', _build.has_scripts),
        ('build_widget',  no_widget_option_is_not_set)
    ]


class bdist(_bdist):
    extra_options_classes = [HNSWOptions, WidgetOptions, BuildExtOptions]

    user_options = _bdist.user_options + OptionsHelper.get_user_options(extra_options_classes)

    def initialize_options(self):
        _bdist.initialize_options(self)
        OptionsHelper.initialize_options(self)

    def finalize_options(self):
        _bdist.finalize_options(self)
        OptionsHelper.finalize_options(self)

    def run(self):
        OptionsHelper.propagate(
            self,
            "build",
            HNSWOptions.get_options_attribute_names()
            + WidgetOptions.get_options_attribute_names()
            + BuildExtOptions.get_options_attribute_names()
        )
        _bdist.run(self)


class bdist_wheel(_bdist_wheel):
    extra_options_classes = [HNSWOptions, WidgetOptions, BuildExtOptions]

    user_options = _bdist_wheel.user_options + OptionsHelper.get_user_options(extra_options_classes)

    def initialize_options(self):
        _bdist_wheel.initialize_options(self)
        OptionsHelper.initialize_options(self)

    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        OptionsHelper.finalize_options(self)

    def run(self):
        OptionsHelper.propagate(
            self,
            "build",
            HNSWOptions.get_options_attribute_names()
            + WidgetOptions.get_options_attribute_names()
            + BuildExtOptions.get_options_attribute_names()
        )
        OptionsHelper.propagate(
            self,
            "install",
            WidgetOptions.get_options_attribute_names()
        )
        _bdist_wheel.run(self)


class build_ext(_build_ext):
    extra_options_classes = [HNSWOptions, BuildExtOptions]

    user_options = _build_ext.user_options +  OptionsHelper.get_user_options(extra_options_classes)

    def initialize_options(self):
        _build_ext.initialize_options(self)
        OptionsHelper.initialize_options(self)

    def finalize_options(self):
        _build_ext.finalize_options(self)
        OptionsHelper.finalize_options(self)

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
            build_type='Debug' if self.debug else 'Release',
            verbose=verbose,
            dry_run=dry_run,
            parallel_build_jobs=self.parallel,
            have_cuda=bool(self.with_cuda),
            cuda_root_dir=self.with_cuda,
            macos_universal_binaries=self.macos_universal_binaries,
            cmake_extra_args=[f'-DPython3_ROOT_DIR={python3_root_dir}']
        )

        if not dry_run:
            logging.info(f'Successfully built {",".join(targets)} {cuda_support_msg}')

    def copy_artifacts_built_with_cmake(self, build_dir, verbose, dry_run):
        for ext in self.extensions:
            # TODO(akhropov): CMake produces wrong artifact names right now so we have to rename it
            src = os.path.join(
                build_dir,
                ext.cmake_build_sub_path,
                build_ext.get_cmake_built_extension_filename(ext.name)
            )
            dst = os.path.join(self.build_lib, ext.dst_sub_path, ext.name + build_ext.get_extension_suffix())
            if dry_run:
                # distutils.file_util.copy_file checks that src file exists so we can't just call it here
                distutils.file_util.log.info(f'copying {src} -> {dst}')
            else:
                distutils.file_util.copy_file(src, dst, verbose=verbose, dry_run=dry_run)


class build_widget(setuptools.Command, setuptools.command.build.SubCommand):
    description = "build CatBoost Jupyter visualization widget (requires yarn (https://yarnpkg.com/))"

    user_options = [
        ('build-generated=', 'b', "directory for built modules"),
    ]

    boolean_options = ['inplace']

    inplace: bool = False

    def initialize_options(self):
        self.editable_mode = False
        self.build_generated = None
        self.inplace = False

    def finalize_options(self):
        self.build_generated = os.path.join('build', 'widget')

        if self.editable_mode:
            self.inplace = True

    def _build(self, verbose, dry_run):
        src_js_dir = os.path.join('catboost', 'widget', 'js')

        distutils.dir_util.copy_tree(
            src_js_dir,
            self.build_generated,
            verbose=verbose,
            dry_run=dry_run
        )

        if not dry_run:
            os.chdir(self.build_generated)
        try:
            for sub_cmd in ['clean', 'install', 'build']:
                spawn(['yarn', sub_cmd], verbose=verbose, dry_run=dry_run)
        finally:
            os.chdir(SETUP_DIR)

    def get_source_files(self):
        result = []
        for dirpath, _, filenames in os.walk(os.path.join('catboost', 'widget', 'js')):
            result += [os.path.join(dirpath, f) for f in filenames]
        return result

    def get_output_mapping(self) -> Dict[str, str]:
        # because they will go 'data' part they won't be returned there
        return {}

    def get_outputs(self) -> List[str]:
        # because they will go 'data' part they won't be returned there
        return []

    def get_data_files(self, dry_run):
        # data_files in the same format as setup's data_files argument
        data_files = []

        src_dir = os.path.join(self.build_generated, 'nbextension')
        dst_dir = os.path.join('share', 'jupyter', 'nbextensions', 'catboost-widget')
        data_files.append( (dst_dir, [os.path.join(src_dir, f) for f in ['extension.js', 'index.js']]) )

        src_dir = os.path.join(self.build_generated, 'labextension')
        dst_dir = os.path.join('share', 'jupyter', 'labextensions', 'catboost-widget')

        data_files.append( (dst_dir, [os.path.join(src_dir, 'package.json')]) )

        src_dir = os.path.join(src_dir, 'static')
        dst_dir = os.path.join(dst_dir, 'static')
        if dry_run and not os.path.exists(src_dir):
            raise RuntimeError("Cannot do dry_run because contents of labextension/static depend on really running build_widget")
        src_files = [ os.path.join(src_dir, f) for f in os.listdir(src_dir) ]
        data_files.append( (dst_dir, src_files) )

        dst_dir = os.path.join('etc', 'jupyter', 'nbconfig', 'notebook.d')
        data_files.append( (dst_dir, [os.path.join('catboost', 'widget', 'catboost-widget.json')]) )

        return data_files

    def run(self):
        verbose = self.distribution.verbose
        dry_run = self.distribution.dry_run

        self._build(verbose, dry_run)

class install(_install):
    extra_options_classes = [WidgetOptions]

    user_options = _install.user_options + OptionsHelper.get_user_options(extra_options_classes)

    def initialize_options(self):
        _install.initialize_options(self)
        OptionsHelper.initialize_options(self)

    def finalize_options(self):
        _install.finalize_options(self)
        OptionsHelper.finalize_options(self)

    def run(self):
        if not self.no_widget:
            if self.distribution.data_files is None:
                self.distribution.data_files = []
            self.distribution.data_files += self.get_finalized_command("build_widget").get_data_files(
                dry_run=self.distribution.dry_run
            )

        _install.run(self)

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
    setup_hnsw_submodule(sys.argv, extensions)

    setup_requires = get_setup_requires(sys.argv)

    setup(
        name=os.environ.get('CATBOOST_PACKAGE_NAME') or 'catboost',
        version=os.environ.get('CATBOOST_PACKAGE_VERSION') or guess_catboost_version(),
        packages=find_packages(),
        package_data={
            'catboost.widget': ['__init__.py', 'ipythonwidget.py', 'metrics_plotter.py', 'callbacks.py'],
        },
        ext_modules=extensions,
        cmdclass={
            'bdist_wheel': bdist_wheel,
            'bdist': bdist,
            'build_ext': build_ext,
            'build_widget': build_widget,
            'build': build,
            'install': install,
            'sdist': sdist,
        },
        extras_require={
            # TODO: fix for ipywidgets >= 8.0 (https://github.com/catboost/catboost/issues/2266)
            'widget': ['ipywidgets (>=7.0, <8.0)']
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
        setup_requires=setup_requires,
    )
