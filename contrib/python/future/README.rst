.. _overview:

Overview: Easy, clean, reliable Python 2/3 compatibility
========================================================

.. image:: https://travis-ci.org/PythonCharmers/python-future.svg?branch=master
    :target: https://travis-ci.org/PythonCharmers/python-future

.. image:: https://readthedocs.org/projects/python-future/badge/?version=latest
    :target: https://python-future.readthedocs.io/en/latest/?badge=latest

``python-future`` is the missing compatibility layer between Python 2 and
Python 3. It allows you to use a single, clean Python 3.x-compatible
codebase to support both Python 2 and Python 3 with minimal overhead.

It provides ``future`` and ``past`` packages with backports and forward
ports of features from Python 3 and 2. It also comes with ``futurize`` and
``pasteurize``, customized 2to3-based scripts that helps you to convert
either Py2 or Py3 code easily to support both Python 2 and 3 in a single
clean Py3-style codebase, module by module.

Notable projects that use ``python-future`` for Python 2/3 compatibility
are `Mezzanine <http://mezzanine.jupo.org/>`_ and `ObsPy
<http://obspy.org>`_.

.. _features:

Features
--------

-   ``future.builtins`` package (also available as ``builtins`` on Py2) provides
    backports and remappings for 20 builtins with different semantics on Py3
    versus Py2

-   support for directly importing 30 standard library modules under
    their Python 3 names on Py2

-   support for importing the other 14 refactored standard library modules
    under their Py3 names relatively cleanly via
    ``future.standard_library`` and ``future.moves``

-   ``past.builtins`` package provides forward-ports of 19 Python 2 types and
    builtin functions. These can aid with per-module code migrations.

-   ``past.translation`` package supports transparent translation of Python 2
    modules to Python 3 upon import. [This feature is currently in alpha.]

-   1000+ unit tests, including many from the Py3.3 source tree.

-   ``futurize`` and ``pasteurize`` scripts based on ``2to3`` and parts of
    ``3to2`` and ``python-modernize``, for automatic conversion from either Py2
    or Py3 to a clean single-source codebase compatible with Python 2.6+ and
    Python 3.3+.

-   a curated set of utility functions and decorators in ``future.utils`` and
    ``past.utils`` selected from Py2/3 compatibility interfaces from projects
    like ``six``, ``IPython``, ``Jinja2``, ``Django``, and ``Pandas``.

-   support for the ``surrogateescape`` error handler when encoding and
    decoding the backported ``str`` and ``bytes`` objects. [This feature is
    currently in alpha.]

-   support for pre-commit hooks

.. _code-examples:

Code examples
-------------

Replacements for Py2's built-in functions and types are designed to be imported
at the top of each Python module together with Python's built-in ``__future__``
statements. For example, this code behaves identically on Python 2.6/2.7 after
these imports as it does on Python 3.3+:

.. code-block:: python

    from __future__ import absolute_import, division, print_function
    from builtins import (bytes, str, open, super, range,
                          zip, round, input, int, pow, object)

    # Backported Py3 bytes object
    b = bytes(b'ABCD')
    assert list(b) == [65, 66, 67, 68]
    assert repr(b) == "b'ABCD'"
    # These raise TypeErrors:
    # b + u'EFGH'
    # bytes(b',').join([u'Fred', u'Bill'])

    # Backported Py3 str object
    s = str(u'ABCD')
    assert s != bytes(b'ABCD')
    assert isinstance(s.encode('utf-8'), bytes)
    assert isinstance(b.decode('utf-8'), str)
    assert repr(s) == "'ABCD'"      # consistent repr with Py3 (no u prefix)
    # These raise TypeErrors:
    # bytes(b'B') in s
    # s.find(bytes(b'A'))

    # Extra arguments for the open() function
    f = open('japanese.txt', encoding='utf-8', errors='replace')

    # New zero-argument super() function:
    class VerboseList(list):
        def append(self, item):
            print('Adding an item')
            super().append(item)

    # New iterable range object with slicing support
    for i in range(10**15)[:10]:
        pass

    # Other iterators: map, zip, filter
    my_iter = zip(range(3), ['a', 'b', 'c'])
    assert my_iter != list(my_iter)

    # The round() function behaves as it does in Python 3, using
    # "Banker's Rounding" to the nearest even last digit:
    assert round(0.1250, 2) == 0.12

    # input() replaces Py2's raw_input() (with no eval()):
    name = input('What is your name? ')
    print('Hello ' + name)

    # pow() supports fractional exponents of negative numbers like in Py3:
    z = pow(-1, 0.5)

    # Compatible output from isinstance() across Py2/3:
    assert isinstance(2**64, int)        # long integers
    assert isinstance(u'blah', str)
    assert isinstance('blah', str)       # only if unicode_literals is in effect

    # Py3-style iterators written as new-style classes (subclasses of
    # future.types.newobject) are automatically backward compatible with Py2:
    class Upper(object):
        def __init__(self, iterable):
            self._iter = iter(iterable)
        def __next__(self):                 # note the Py3 interface
            return next(self._iter).upper()
        def __iter__(self):
            return self
    assert list(Upper('hello')) == list('HELLO')


There is also support for renamed standard library modules. The recommended
interface works like this:

.. code-block:: python

    # Many Py3 module names are supported directly on both Py2.x and 3.x:
    from http.client import HttpConnection
    import html.parser
    import queue
    import xmlrpc.client

    # Refactored modules with clashing names on Py2 and Py3 are supported
    # as follows:
    from future import standard_library
    standard_library.install_aliases()

    # Then, for example:
    from itertools import filterfalse, zip_longest
    from urllib.request import urlopen
    from collections import ChainMap
    from collections import UserDict, UserList, UserString
    from subprocess import getoutput, getstatusoutput
    from collections import Counter, OrderedDict   # backported to Py2.6


Automatic conversion to Py2/3-compatible code
---------------------------------------------

``python-future`` comes with two scripts called ``futurize`` and
``pasteurize`` to aid in making Python 2 code or Python 3 code compatible with
both platforms (Py2/3). It is based on 2to3 and uses fixers from ``lib2to3``,
``lib3to2``, and ``python-modernize``, as well as custom fixers.

``futurize`` passes Python 2 code through all the appropriate fixers to turn it
into valid Python 3 code, and then adds ``__future__`` and ``future`` package
imports so that it also runs under Python 2.

For conversions from Python 3 code to Py2/3, use the ``pasteurize`` script
instead. This converts Py3-only constructs (e.g. new metaclass syntax) to
Py2/3 compatible constructs and adds ``__future__`` and ``future`` imports to
the top of each module.

In both cases, the result should be relatively clean Py3-style code that runs
mostly unchanged on both Python 2 and Python 3.

Futurize: 2 to both
~~~~~~~~~~~~~~~~~~~

For example, running ``futurize -w mymodule.py`` turns this Python 2 code:

.. code-block:: python

    import Queue
    from urllib2 import urlopen

    def greet(name):
        print 'Hello',
        print name

    print "What's your name?",
    name = raw_input()
    greet(name)

into this code which runs on both Py2 and Py3:

.. code-block:: python

    from __future__ import print_function
    from future import standard_library
    standard_library.install_aliases()
    from builtins import input
    import queue
    from urllib.request import urlopen

    def greet(name):
        print('Hello', end=' ')
        print(name)

    print("What's your name?", end=' ')
    name = input()
    greet(name)

See :ref:`forwards-conversion` and :ref:`backwards-conversion` for more details.


Automatic translation
---------------------

The ``past`` package can automatically translate some simple Python 2
modules to Python 3 upon import. The goal is to support the "long tail" of
real-world Python 2 modules (e.g. on PyPI) that have not been ported yet. For
example, here is how to use a Python 2-only package called ``plotrique`` on
Python 3. First install it:

.. code-block:: bash

    $ pip3 install plotrique==0.2.5-7 --no-compile   # to ignore SyntaxErrors

(or use ``pip`` if this points to your Py3 environment.)

Then pass a whitelist of module name prefixes to the ``autotranslate()`` function.
Example:

.. code-block:: bash

    $ python3

    >>> from past.translation import autotranslate
    >>> autotranslate(['plotrique'])
    >>> import plotrique

This transparently translates and runs the ``plotrique`` module and any
submodules in the ``plotrique`` package that ``plotrique`` imports.

This is intended to help you migrate to Python 3 without the need for all
your code's dependencies to support Python 3 yet. It should be used as a
last resort; ideally Python 2-only dependencies should be ported
properly to a Python 2/3 compatible codebase using a tool like
``futurize`` and the changes should be pushed to the upstream project.

Note: the auto-translation feature is still in alpha; it needs more testing and
development, and will likely never be perfect.

For more info, see :ref:`translation`.

Pre-commit hooks
----------------

`Pre-commit <https://pre-commit.com/>`_ is a framework for managing and maintaining
multi-language pre-commit hooks.

In case you need to port your project from Python 2 to Python 3, you might consider
using such hook during the transition period.

First:

.. code-block:: bash

    $ pip install pre-commit

and then in your project's directory:

.. code-block:: bash

    $ pre-commit install

Next, you need to add this entry to your ``.pre-commit-config.yaml``

.. code-block:: yaml

    -   repo: https://github.com/PythonCharmers/python-future
        rev: master
        hooks:
            - id: futurize
              args: [--both-stages]

The ``args`` part is optional, by default only stage1 is applied.

Licensing
---------

:Author:  Ed Schofield, Jordan M. Adler, et al

:Copyright: 2013-2019 Python Charmers Pty Ltd, Australia.

:Sponsors: Python Charmers Pty Ltd, Australia, and Python Charmers Pte
           Ltd, Singapore. http://pythoncharmers.com

           Pinterest https://opensource.pinterest.com/

:Licence: MIT. See ``LICENSE.txt`` or `here <http://python-future.org/credits.html>`_.

:Other credits:  See `here <http://python-future.org/credits.html>`_.


Next steps
----------

If you are new to Python-Future, check out the `Quickstart Guide
<http://python-future.org/quickstart.html>`_.

For an update on changes in the latest version, see the `What's New
<http://python-future.org/whatsnew.html>`_ page.
