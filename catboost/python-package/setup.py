from setuptools import setup, find_packages

PACKAGE = 'catboost'

setup(
    name=PACKAGE,
    version="0.13",

    author="CatBoost Developers",
    description="Python package for catboost",
    license="Apache License, Version 2.0",

    packages=find_packages(),
    install_requires=[
        'enum34',
        'graphviz',
        'six',
        'numpy >= 1.11.1',
        'pandas >= 0.19.1'
    ],
)
