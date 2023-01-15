# -*- coding: utf-8 -*-
from setuptools import find_packages, setup


setup(
    name='minidump2core',
    version='0.0.6',
    packages=find_packages('src'),
    package_dir={'minidump2core': 'src/minidump2core'},
    url='https://wiki.yandex-team.ru/Development/Poisk/arcadia/devtools/coredump_filter/',
    license='Yandex',
    author='Mikhail Veltishchev',
    author_email='mvel@yandex-team.ru',
    description='Parsed minidump to GDB traces converter',
)
