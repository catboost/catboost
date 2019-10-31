from setuptools import setup, find_packages
import sys


PACKAGE = 'catboost'

def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)

def version():
    globals={}
    execfile('./catboost/version.py', globals)
    return globals['VERSION']

install_requires=[
    'graphviz',
    'matplotlib',
    'numpy >= 1.16.0',
    'pandas >= 0.24.0',
    'scipy',
    'plotly',
    'six',
    'enum34; python_version < "3.4"'
],

setup(
    name=PACKAGE,
    version=version(),

    author="CatBoost Developers",
    description="Python package for catboost",
    license="Apache License, Version 2.0",

    packages=find_packages(),
    install_requires=install_requires,
)
