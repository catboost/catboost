PyParsing -- A Python Parsing Module
====================================

|Version| |Build Status| |Coverage| |License| |Python Versions| |Snyk Score|

Introduction
============

The pyparsing module is an alternative approach to creating and
executing simple grammars, vs. the traditional lex/yacc approach, or the
use of regular expressions. The pyparsing module provides a library of
classes that client code uses to construct the grammar directly in
Python code.

*[Since first writing this description of pyparsing in late 2003, this
technique for developing parsers has become more widespread, under the
name Parsing Expression Grammars - PEGs. See more information on PEGs*
`here <https://en.wikipedia.org/wiki/Parsing_expression_grammar>`__
*.]*

Here is a program to parse ``"Hello, World!"`` (or any greeting of the form
``"salutation, addressee!"``):

.. code:: python

    from pyparsing import Word, alphas
    greet = Word(alphas) + "," + Word(alphas) + "!"
    hello = "Hello, World!"
    print(hello, "->", greet.parseString(hello))

The program outputs the following::

    Hello, World! -> ['Hello', ',', 'World', '!']

The Python representation of the grammar is quite readable, owing to the
self-explanatory class names, and the use of '+', '|' and '^' operator
definitions.

The parsed results returned from ``parseString()`` is a collection of type
``ParseResults``, which can be accessed as a
nested list, a dictionary, or an object with named attributes.

The pyparsing module handles some of the problems that are typically
vexing when writing text parsers:

- extra or missing whitespace (the above program will also handle ``"Hello,World!"``, ``"Hello , World !"``, etc.)
- quoted strings
- embedded comments

The examples directory includes a simple SQL parser, simple CORBA IDL
parser, a config file parser, a chemical formula parser, and a four-
function algebraic notation parser, among many others.

Documentation
=============

There are many examples in the online docstrings of the classes
and methods in pyparsing. You can find them compiled into `online docs <https://pyparsing-docs.readthedocs.io/en/latest/>`__. Additional
documentation resources and project info are listed in the online
`GitHub wiki <https://github.com/pyparsing/pyparsing/wiki>`__. An
entire directory of examples can be found `here <https://github.com/pyparsing/pyparsing/tree/master/examples>`__.

License
=======

MIT License. See header of the `pyparsing __init__.py <https://github.com/pyparsing/pyparsing/blob/master/pyparsing/__init__.py#L1-L23>`__ file.

History
=======

See `CHANGES <https://github.com/pyparsing/pyparsing/blob/master/CHANGES>`__ file.

.. |Build Status| image:: https://github.com/pyparsing/pyparsing/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/pyparsing/pyparsing/actions/workflows/ci.yml

.. |Coverage| image:: https://codecov.io/gh/pyparsing/pyparsing/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pyparsing/pyparsing

.. |Version| image:: https://img.shields.io/pypi/v/pyparsing?style=flat-square
    :target: https://pypi.org/project/pyparsing/
    :alt: Version

.. |License| image:: https://img.shields.io/pypi/l/pyparsing.svg?style=flat-square
    :target: https://pypi.org/project/pyparsing/
    :alt: License

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pyparsing.svg?style=flat-square
    :target: https://pypi.org/project/python-liquid/
    :alt: Python versions

.. |Snyk Score| image:: https://snyk.io//advisor/python/pyparsing/badge.svg
   :target: https://snyk.io//advisor/python/pyparsing
   :alt: pyparsing
