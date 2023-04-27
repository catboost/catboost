pytest-asyncio
==============

.. image:: https://img.shields.io/pypi/v/pytest-asyncio.svg
    :target: https://pypi.python.org/pypi/pytest-asyncio
.. image:: https://github.com/pytest-dev/pytest-asyncio/workflows/CI/badge.svg
    :target: https://github.com/pytest-dev/pytest-asyncio/actions?workflow=CI
.. image:: https://codecov.io/gh/pytest-dev/pytest-asyncio/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytest-dev/pytest-asyncio
.. image:: https://img.shields.io/pypi/pyversions/pytest-asyncio.svg
    :target: https://github.com/pytest-dev/pytest-asyncio
    :alt: Supported Python versions
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

`pytest-asyncio <https://pytest-asyncio.readthedocs.io/en/latest/>`_ is a `pytest <https://docs.pytest.org/en/latest/contents.html>`_ plugin. It facilitates testing of code that uses the `asyncio <https://docs.python.org/3/library/asyncio.html>`_ library.

Specifically, pytest-asyncio provides support for coroutines as test functions. This allows users to *await* code inside their tests. For example, the following code is executed as a test item by pytest:

.. code-block:: python

    @pytest.mark.asyncio
    async def test_some_asyncio_code():
        res = await library.do_something()
        assert b"expected result" == res

More details can be found in the `documentation <https://pytest-asyncio.readthedocs.io/en/latest/>`_.

Note that test classes subclassing the standard `unittest <https://docs.python.org/3/library/unittest.html>`__ library are not supported. Users
are advised to use `unittest.IsolatedAsyncioTestCase <https://docs.python.org/3/library/unittest.html#unittest.IsolatedAsyncioTestCase>`__
or an async framework such as `asynctest <https://asynctest.readthedocs.io/en/latest>`__.


pytest-asyncio is available under the `Apache License 2.0 <https://github.com/pytest-dev/pytest-asyncio/blob/master/LICENSE>`_.


Installation
------------

To install pytest-asyncio, simply:

.. code-block:: bash

    $ pip install pytest-asyncio

This is enough for pytest to pick up pytest-asyncio.


Contributing
------------
Contributions are very welcome. Tests can be run with ``tox``, please ensure
the coverage at least stays the same before you submit a pull request.
