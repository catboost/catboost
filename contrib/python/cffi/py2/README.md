CFFI
====

Foreign Function Interface for Python calling C code.
Please see the [Documentation](http://cffi.readthedocs.org/) or uncompiled
in the doc/ subdirectory.

Download
--------

[Download page](https://foss.heptapod.net/pypy/cffi/-/tags)

Contact
-------

[Mailing list](https://groups.google.com/forum/#!forum/python-cffi)

Testing/development tips
------------------------

To run tests under CPython, run::

    pip install pytest     # if you don't have py.test already
    pip install pycparser
    python setup.py build_ext -f -i
    py.test c/ testing/

If you run in another directory (either the tests or another program),
you should use the environment variable ``PYTHONPATH=/path`` to point
to the location that contains the ``_cffi_backend.so`` just compiled.
