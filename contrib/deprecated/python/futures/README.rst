.. image:: https://travis-ci.org/agronholm/pythonfutures.svg?branch=master
  :target: https://travis-ci.org/agronholm/pythonfutures
  :alt: Build Status

This is a backport of the `concurrent.futures`_ standard library module to Python 2.

It **does not** work on Python 3 due to Python 2 syntax being used in the codebase.
Python 3 users should not attempt to install it, since the package is already included in the
standard library.

To conditionally require this library only on Python 2, you can do this in your ``setup.py``:

.. code-block:: python

    setup(
        ...
        extras_require={
            ':python_version == "2.7"': ['futures']
        }
    )

Or, using the newer syntax:

.. code-block:: python

    setup(
        ...
        install_requires={
            'futures; python_version == "2.7"'
        }
    )

.. warning:: The ``ProcessPoolExecutor`` class has known (unfixable) problems on Python 2 and
   should not be relied on for mission critical work. Please see `Issue 29 <https://github.com/agronholm/pythonfutures/issues/29>`_ and `upstream bug report <https://bugs.python.org/issue9205>`_ for more details.

.. _concurrent.futures: https://docs.python.org/library/concurrent.futures.html
