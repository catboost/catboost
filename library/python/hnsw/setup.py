from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.dir_util import mkpath
from distutils.file_util import copy_file


class CopyBuild(build_ext):
    def build_extension(self, ext):
        mkpath(self.build_lib)
        copy_file(ext.sources[0], self.get_ext_fullpath(ext.name))

# How to build and upload package to Yandex PyPI can be found here: https://wiki.yandex-team.ru/pypi/
# Before building and uploading _hnsw.so should be built from 'hnsw' folder like this:
# ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=python2-config
# Run setup.py with python which config you used for building _hnsw.so on previous step.

setup(
    name='hnsw',
    version='0.2.1',
    description='Python wrapper for Hierarchical Navigable Small World index implementation',
    author='Ivan Lyzhin',
    author_email='ilyzhin@yandex-team.ru',
    packages=['hnsw'],
    cmdclass={'build_ext': CopyBuild},
    ext_modules=[Extension('hnsw/_hnsw', ['hnsw/_hnsw.so'])]
)
