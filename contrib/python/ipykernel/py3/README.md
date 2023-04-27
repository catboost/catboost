# IPython Kernel for Jupyter

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
