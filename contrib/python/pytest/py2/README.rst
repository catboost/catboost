.. image:: https://docs.pytest.org/en/latest/_static/pytest1.png
   :target: https://docs.pytest.org/en/latest/
   :align: center
   :alt: pytest


------

.. image:: https://img.shields.io/pypi/v/pytest.svg
    :target: https://pypi.org/project/pytest/

.. image:: https://img.shields.io/conda/vn/conda-forge/pytest.svg
    :target: https://anaconda.org/conda-forge/pytest

.. image:: https://img.shields.io/pypi/pyversions/pytest.svg
    :target: https://pypi.org/project/pytest/

.. image:: https://codecov.io/gh/pytest-dev/pytest/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytest-dev/pytest
    :alt: Code coverage Status

.. image:: https://travis-ci.org/pytest-dev/pytest.svg?branch=master
    :target: https://travis-ci.org/pytest-dev/pytest

.. image:: https://dev.azure.com/pytest-dev/pytest/_apis/build/status/pytest-CI?branchName=master
    :target: https://dev.azure.com/pytest-dev/pytest

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black

.. image:: https://www.codetriage.com/pytest-dev/pytest/badges/users.svg
    :target: https://www.codetriage.com/pytest-dev/pytest

The ``pytest`` framework makes it easy to write small tests, yet
scales to support complex functional testing for applications and libraries.

An example of a simple test:

.. code-block:: python

    # content of test_sample.py
    def inc(x):
        return x + 1


    def test_answer():
        assert inc(3) == 5


To execute it::

    $ pytest
    ============================= test session starts =============================
    collected 1 items

    test_sample.py F

    ================================== FAILURES ===================================
    _________________________________ test_answer _________________________________

        def test_answer():
    >       assert inc(3) == 5
    E       assert 4 == 5
    E        +  where 4 = inc(3)

    test_sample.py:5: AssertionError
    ========================== 1 failed in 0.04 seconds ===========================


Due to ``pytest``'s detailed assertion introspection, only plain ``assert`` statements are used. See `getting-started <https://docs.pytest.org/en/latest/getting-started.html#our-first-test-run>`_ for more examples.


Features
--------

- Detailed info on failing `assert statements <https://docs.pytest.org/en/latest/assert.html>`_ (no need to remember ``self.assert*`` names);

- `Auto-discovery
  <https://docs.pytest.org/en/latest/goodpractices.html#python-test-discovery>`_
  of test modules and functions;

- `Modular fixtures <https://docs.pytest.org/en/latest/fixture.html>`_ for
  managing small or parametrized long-lived test resources;

- Can run `unittest <https://docs.pytest.org/en/latest/unittest.html>`_ (or trial),
  `nose <https://docs.pytest.org/en/latest/nose.html>`_ test suites out of the box;

- Python 2.7, Python 3.4+, PyPy 2.3, Jython 2.5 (untested);

- Rich plugin architecture, with over 315+ `external plugins <http://plugincompat.herokuapp.com>`_ and thriving community;


Documentation
-------------

For full documentation, including installation, tutorials and PDF documents, please see https://docs.pytest.org/en/latest/.


Bugs/Requests
-------------

Please use the `GitHub issue tracker <https://github.com/pytest-dev/pytest/issues>`_ to submit bugs or request features.


Changelog
---------

Consult the `Changelog <https://docs.pytest.org/en/latest/changelog.html>`__ page for fixes and enhancements of each version.


Support pytest
--------------

You can support pytest by obtaining a `Tideflift subscription`_.

Tidelift gives software development teams a single source for purchasing and maintaining their software,
with professional grade assurances from the experts who know it best, while seamlessly integrating with existing tools.


.. _`Tideflift subscription`: https://tidelift.com/subscription/pkg/pypi-pytest?utm_source=pypi-pytest&utm_medium=referral&utm_campaign=readme


Security
^^^^^^^^

pytest has never been associated with a security vunerability, but in any case, to report a
security vulnerability please use the `Tidelift security contact <https://tidelift.com/security>`_.
Tidelift will coordinate the fix and disclosure.


License
-------

Copyright Holger Krekel and others, 2004-2020.

Distributed under the terms of the `MIT`_ license, pytest is free and open source software.

.. _`MIT`: https://github.com/pytest-dev/pytest/blob/master/LICENSE
