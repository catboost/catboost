.. image:: https://img.shields.io/pypi/v/singledispatch.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/singledispatch.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/singledispatch

.. image:: https://github.com/jaraco/singledispatch/workflows/tests/badge.svg
   :target: https://github.com/jaraco/singledispatch/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2021-informational
   :target: https://blog.jaraco.com/skeleton

`PEP 443 <http://www.python.org/dev/peps/pep-0443/>`_ proposed to expose
a mechanism in the ``functools`` standard library module in Python 3.4
that provides a simple form of generic programming known as
single-dispatch generic functions.

This library is a backport of this functionality and its evolution.

Refer to the `upstream documentation
<http://docs.python.org/3/library/functools.html#functools.singledispatch>`_
for API guidance. To use the backport, simply use
``from singledispatch import singledispatch, singledispatchmethod`` in place of
``from functools import singledispatch, singledispatchmethod``.



Maintenance
-----------

This backport is maintained on Github by Jason R. Coombs, one of the
members of the core CPython team:

* `repository <https://github.com/jaraco/singledispatch>`_

* `issue tracker <https://github.com/jaraco/singledispatch/issues>`_
