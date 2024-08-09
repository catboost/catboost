[![GitHub Actions Status](https://github.com/python-cffi/cffi/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/python-cffi/cffi/actions/workflows/ci.yaml?query=branch%3Amain++)
[![PyPI version](https://img.shields.io/pypi/v/cffi.svg)](https://pypi.org/project/cffi)
[![Read the Docs](https://img.shields.io/badge/docs-latest-blue.svg)][Documentation]


CFFI
====

Foreign Function Interface for Python calling C code.

Please see the [Documentation] or uncompiled in the `doc/` subdirectory.

Download
--------

[Download page](https://github.com/python-cffi/cffi/releases)

Source Code
-----------

Source code is publicly available on
[GitHub](https://github.com/python-cffi/cffi).

Contact
-------

[Mailing list](https://groups.google.com/forum/#!forum/python-cffi)

Testing/development tips
------------------------

To run tests under CPython, run the following in the source root directory:

    pip install pytest
    pip install -e .  # editable install of CFFI for local development
    pytest c/ testing/


[Documentation]: http://cffi.readthedocs.org/
