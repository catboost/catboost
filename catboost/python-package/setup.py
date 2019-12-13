import distutils
import logging
import os
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


def copy_catboost_sources(topdir, pkgdir, verbose, dry_run):
    topnames = (
        '.arcadia.root',
        'AUTHORS', 'LICENSE', 'CONTRIBUTING.md', 'README.md', 'RELEASE.md',
        'ya', 'ya.bat', 'ya.conf', 'ya.make',
        'build',
        'catboost/cuda',
        'catboost/idl',
        'catboost/libs',
        'catboost/private',
        'catboost/python-package/catboost',
        'contrib/libs',
        'contrib/python',
        'contrib/tools',
        'library',
        'make',
        'msvs',
        'tools',
        'util',
    )
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
        ('python-config=', None, emph('Python configure script, e.g. python3-config')),
        ('with-cuda=', None, emph('Build with CUDA support (cuda-root|no)')),
        ('os-sdk=', None, emph('For Yandex-internal use, e.g. local')),
        ('with-ymake=', None, emph('Use ymake or not (YES|no)')),
        ('parallel=', 'j', emph('Number of parallel build jobs')),
    ]

    def initialize_options(self):
        self.python_config = 'python3-config' if python_version().startswith('3') else 'python-config'
        self.with_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_ROOT') or None
        self.os_sdk = 'local'
        self.with_ymake = True
        self.parallel = None

    def finalize_options(self):
        if os.path.exists(str(self.with_cuda)):
            logging.info("Targeting for CUDA support with {}".format(self.with_cuda))
        else:
            self.with_cuda = None
        self.with_ymake = self.with_ymake not in ('N', 'n', 'No', 'no', 'None', 'none', False, 'False', 'false', 'F', 'f', '0', None)

    def propagate_options(self, origin, subcommand):
        self.distribution.get_command_obj(subcommand).set_undefined_options(
            origin,
            ("python_config", "python_config"),
            ("with_cuda", "with_cuda"),
            ("os_sdk", "os_sdk"),
            ("with_ymake", "with_ymake"),
            ("parallel", "parallel"),
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


class build_ext(_build_ext):

    user_options = _build_ext.user_options + Helper.options

    def initialize_options(self):
        _build_ext.initialize_options(self)
        Helper.initialize_options(self)

    def finalize_options(self):
        _build_ext.finalize_options(self)
        Helper.finalize_options(self)

    def run(self):
        if os.path.exists(os.path.join(SETUP_DIR, PKG_INFO)):
            topsrc_dir = os.path.join(SETUP_DIR, EXT_SRC)
        else:
            topsrc_dir = os.path.join(SETUP_DIR, '..', '..')

        catboost_ext = {
            'linux': '_catboost.so',
            'darwin': '_catboost.dylib',
            'win32': '_catboost.pyd',
        }[sys.platform]

        build_dir = os.path.abspath(self.build_temp)
        put_dir = os.path.abspath(os.path.join(self.build_lib, 'catboost'))

        verbose = self.distribution.verbose
        dry_run = self.distribution.dry_run
        distutils.dir_util.mkpath(put_dir, verbose=verbose, dry_run=dry_run)

        if self.with_ymake:
            self.build_with_ymake(topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run)
        elif 'win' in sys.platform:
            self.build_with_msbuild(topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run)
        else:
            self.build_with_make(topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run)

    def build_with_ymake(self, topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run):
        logging.info('Buildling {} with ymake'.format(catboost_ext))
        ya = os.path.abspath(os.path.join(topsrc_dir, 'ya'))
        python = sys.executable
        ymake_cmd = [
            python, ya, 'make', os.path.join(topsrc_dir, 'catboost', 'python-package', 'catboost'),
            '--no-src-links',
            '--output', build_dir,
            '-DPYTHON_CONFIG=' + self.python_config,
            '-DUSE_ARCADIA_PYTHON=no',
            '-DOS_SDK=' + self.os_sdk,
        ]
        ymake_cmd += ['-d'] if self.debug else ['-r', '-DNO_DEBUGINFO']
        if self.os_sdk != 'local':
            ymake_cmd += ['-DUSE_SYSTEM_PYTHON=' + '.'.join(python_version().split('.')[:2])]
        if self.parallel is not None:
            ymake_cmd += ['-j', str(self.parallel)]
        dll = os.path.join(build_dir, 'catboost', 'python-package', 'catboost', catboost_ext)
        if self.with_cuda:
            try:
                logging_execute(ymake_cmd + ['-DHAVE_CUDA=yes', '-DCUDA_ROOT={}'.format(self.with_cuda)], verbose, dry_run)
                logging.info('Successfully built {} with CUDA support'.format(catboost_ext))
                distutils.file_util.copy_file(dll, put_dir, verbose=verbose, dry_run=dry_run)
                return
            except subprocess.CalledProcessError:
                logging.error('Cannot build {} with CUDA support, will build without CUDA'.format(catboost_ext))

        logging_execute(ymake_cmd + ['-DHAVE_CUDA=no'], verbose, dry_run)
        logging.info('Successfully built {} without CUDA support'.format(catboost_ext))
        distutils.file_util.copy_file(dll, put_dir, verbose=verbose, dry_run=dry_run)

    def build_with_make(self, topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run):
        logging.info('Buildling {} with gnu make'.format(catboost_ext))
        makefile = 'python{}.{}CLANG50-LINUX-X86_64.makefile'.format(python_version()[0], 'CUDA.' if self.with_cuda else '')
        make_cmd = [
            'make', '-f', '../../make/' + makefile,
            'CC=clang-5.0',
            'CXX=clang++-5.0',
            'BUILD_ROOT=' + build_dir,
            'SOURCE_ROOT=' + topsrc_dir,
        ]
        if self.parallel is not None:
            make_cmd += ['-j', str(self.parallel)]
        logging_execute(make_cmd, verbose, dry_run)
        logging.info('Successfully built {} with{} CUDA support'.format(catboost_ext, '' if self.with_cuda else 'out'))
        dll = os.path.join(build_dir, 'catboost', 'python-package', 'catboost', catboost_ext)
        distutils.file_util.copy_file(dll, put_dir, verbose=verbose, dry_run=dry_run)

    def build_with_msbuild(self, topsrc_dir, build_dir, catboost_ext, put_dir, verbose, dry_run):
        logging.info('Buildling {} with msbuild'.format(catboost_ext))
        raise ValueError('TODO: build with msbuild')


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
        ext_modules=[Extension('_catboost', sources=[])],
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
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "License :: OSI Approved :: Apache License 2.0",
        ],
        install_requires=[
            'graphviz',
            'matplotlib (<3.0); python_version == "2.7"',
            'matplotlib; python_version >= "3.5"',
            'numpy (>=1.16.0,<1.17.0); python_version == "2.7"',
            'numpy (>=1.16.0); python_version >= "3.5"',
            'pandas (>=0.24,<0.25); python_version == "2.7"',
            'pandas (>=0.24); python_version >= "3.5"',
            'scipy (<1.3.0); python_version == "2.7"',
            'scipy; python_version >= "3.5"',
            'plotly',
            'six',
            'enum34; python_version < "3.4"',
        ],
        zip_safe=False,
        setup_requires=['wheel'],
    )
