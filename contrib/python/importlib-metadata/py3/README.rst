.. image:: https://img.shields.io/pypi/v/importlib_metadata.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/importlib_metadata.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/importlib_metadata

.. image:: https://github.com/python/importlib_metadata/workflows/tests/badge.svg
   :target: https://github.com/python/importlib_metadata/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. image:: https://readthedocs.org/projects/importlib-metadata/badge/?version=latest
   :target: https://importlib-metadata.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton


Library to access the metadata for a Python package.

This package supplies third-party access to the functionality of
`importlib.metadata <https://docs.python.org/3/library/importlib.metadata.html>`_
including improvements added to subsequent Python versions.


Compatibility
=============

New features are introduced in this third-party library and later merged
into CPython. The following table indicates which versions of this library
were contributed to different versions in the standard library:

.. list-table::
   :header-rows: 1

   * - importlib_metadata
     - stdlib
   * - 4.8
     - 3.11
   * - 4.4
     - 3.10
   * - 1.4
     - 3.8


Usage
=====

See the `online documentation <https://importlib_metadata.readthedocs.io/>`_
for usage details.

`Finder authors
<https://docs.python.org/3/reference/import.html#finders-and-loaders>`_ can
also add support for custom package installers.  See the above documentation
for details.


Caveats
=======

This project primarily supports third-party packages installed by PyPA
tools (or other conforming packages). It does not support:

- Packages in the stdlib.
- Packages installed without metadata.

Project details
===============

 * Project home: https://github.com/python/importlib_metadata
 * Report bugs at: https://github.com/python/importlib_metadata/issues
 * Code hosting: https://github.com/python/importlib_metadata
 * Documentation: https://importlib_metadata.readthedocs.io/
