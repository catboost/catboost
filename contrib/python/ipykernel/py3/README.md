# IPython Kernel for Jupyter

[![Build Status](https://github.com/ipython/ipykernel/actions/workflows/ci.yml/badge.svg?query=branch%3Amain++)](https://github.com/ipython/ipykernel/actions/workflows/ci.yml/badge.svg?query=branch%3Amain++)
[![codecov](https://codecov.io/gh/ipython/ipykernel/branch/main/graph/badge.svg?token=SyksDOcIJa)](https://codecov.io/gh/ipython/ipykernel)
[![Documentation Status](https://readthedocs.org/projects/ipython/badge/?version=latest)](http://ipython.readthedocs.io/en/latest/?badge=latest)

This package provides the IPython kernel for Jupyter.

## Installation from source

1. `git clone`
1. `cd ipykernel`
1. `pip install -e ".[test]"`

After that, all normal `ipython` commands will use this newly-installed version of the kernel.

## Running tests

Follow the instructions from `Installation from source`.

and then from the root directory

```bash
pytest ipykernel
```

## Running tests with coverage

Follow the instructions from `Installation from source`.

and then from the root directory

```bash
pytest ipykernel -vv -s --cov ipykernel --cov-branch --cov-report term-missing:skip-covered --durations 10
```
