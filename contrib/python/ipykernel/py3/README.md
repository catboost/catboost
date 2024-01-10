# IPython Kernel for Jupyter

[![Build Status](https://github.com/ipython/ipykernel/actions/workflows/ci.yml/badge.svg?query=branch%3Amain++)](https://github.com/ipython/ipykernel/actions/workflows/ci.yml/badge.svg?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/ipykernel/badge/?version=latest)](http://ipykernel.readthedocs.io/en/latest/?badge=latest)

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
pytest
```

## Running tests with coverage

Follow the instructions from `Installation from source`.

and then from the root directory

```bash
pytest -vv -s --cov ipykernel --cov-branch --cov-report term-missing:skip-covered --durations 10
```

## About the IPython Development Team

The IPython Development Team is the set of all contributors to the IPython project.
This includes all of the IPython subprojects.

The core team that coordinates development on GitHub can be found here:
https://github.com/ipython/.

## Our Copyright Policy

IPython uses a shared copyright model. Each contributor maintains copyright
over their contributions to IPython. But, it is important to note that these
contributions are typically only changes to the repositories. Thus, the IPython
source code, in its entirety is not the copyright of any single person or
institution. Instead, it is the collective copyright of the entire IPython
Development Team. If individual contributors want to maintain a record of what
changes/contributions they have specific copyright on, they should indicate
their copyright in the commit message of the change, when they commit the
change to one of the IPython repositories.

With this in mind, the following banner should be used in any source code file
to indicate the copyright and license terms:

```
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
```
