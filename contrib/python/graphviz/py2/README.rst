Graphviz
========

|PyPI version| |License| |Supported Python| |Format|

|Travis| |Codecov| |Readthedocs-stable| |Readthedocs-latest|

This package facilitates the creation and rendering of graph descriptions in
the DOT_ language of the Graphviz_ graph drawing software (`master repo`_) from
Python.

Create a graph object, assemble the graph by adding nodes and edges, and
retrieve its DOT source code string. Save the source code to a file and render
it with the Graphviz installation of your system.

Use the ``view`` option/method to directly inspect the resulting (PDF, PNG,
SVG, etc.) file with its default application. Graphs can also be rendered
and displayed within `Jupyter notebooks`_ (formerly known as
`IPython notebooks`_, example_, nbviewer_) as well as the `Jupyter Qt Console`_.


Links
-----

- GitHub: https://github.com/xflr6/graphviz
- PyPI: https://pypi.org/project/graphviz/
- Documentation: https://graphviz.readthedocs.io
- Changelog: https://graphviz.readthedocs.io/en/latest/changelog.html
- Issue Tracker: https://github.com/xflr6/graphviz/issues
- Download: https://pypi.org/project/graphviz/#files


Installation
------------

This package runs under Python 2.7, and 3.6+, use pip_ to install:

.. code:: bash

    $ pip install graphviz

To render the generated DOT source code, you also need to install Graphviz
(`download page`_, `installation procedure for Windows`_, `archived versions`_).

Make sure that the directory containing the ``dot`` executable is on your
systems' path.


Quickstart
----------

Create a graph object:

.. code:: python

    >>> from graphviz import Digraph

    >>> dot = Digraph(comment='The Round Table')

    >>> dot  #doctest: +ELLIPSIS
    <graphviz.dot.Digraph object at 0x...>

Add nodes and edges:

.. code:: python

    >>> dot.node('A', 'King Arthur')
    >>> dot.node('B', 'Sir Bedevere the Wise')
    >>> dot.node('L', 'Sir Lancelot the Brave')

    >>> dot.edges(['AB', 'AL'])
    >>> dot.edge('B', 'L', constraint='false')

Check the generated source code:

.. code:: python

    >>> print(dot.source)  # doctest: +NORMALIZE_WHITESPACE
    // The Round Table
    digraph {
        A [label="King Arthur"]
        B [label="Sir Bedevere the Wise"]
        L [label="Sir Lancelot the Brave"]
        A -> B
        A -> L
        B -> L [constraint=false]
    }

Save and render the source code, optionally view the result:

.. code:: python

    >>> dot.render('test-output/round-table.gv', view=True)  # doctest: +SKIP
    'test-output/round-table.gv.pdf'

.. image:: https://raw.github.com/xflr6/graphviz/master/docs/round-table.png
    :align: center


See also
--------

- pygraphviz_ |--| full-blown interface wrapping the Graphviz C library with SWIG
- graphviz-python_ |--| official Python bindings (documentation_)
- pydot_ |--| stable pure-Python approach, requires pyparsing


License
-------

This package is distributed under the `MIT license`_.


.. _pip: https://pip.readthedocs.io
.. _Graphviz:  https://www.graphviz.org
.. _master repo: https://gitlab.com/graphviz/graphviz/
.. _download page: https://www.graphviz.org/download/
.. _installation procedure for Windows: https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224
.. _archived versions: https://www2.graphviz.org/Archive/stable/
.. _DOT: https://www.graphviz.org/doc/info/lang.html
.. _Jupyter notebooks: https://jupyter.org
.. _IPython notebooks: https://ipython.org/notebook.html
.. _example: https://github.com/xflr6/graphviz/blob/master/examples/graphviz-notebook.ipynb
.. _nbviewer: https://nbviewer.jupyter.org/github/xflr6/graphviz/blob/master/examples/notebook.ipynb
.. _Jupyter Qt Console: https://qtconsole.readthedocs.io

.. _pygraphviz: https://pypi.org/project/pygraphviz/
.. _graphviz-python: https://pypi.org/project/graphviz-python/
.. _documentation: https://www.graphviz.org/pdf/gv.3python.pdf
.. _pydot: https://pypi.org/project/pydot/

.. _MIT license: https://opensource.org/licenses/MIT


.. |--| unicode:: U+2013


.. |PyPI version| image:: https://img.shields.io/pypi/v/graphviz.svg
    :target: https://pypi.org/project/graphviz/
    :alt: Latest PyPI Version
.. |License| image:: https://img.shields.io/pypi/l/graphviz.svg
    :target: https://pypi.org/project/graphviz/
    :alt: License
.. |Supported Python| image:: https://img.shields.io/pypi/pyversions/graphviz.svg
    :target: https://pypi.org/project/graphviz/
    :alt: Supported Python Versions
.. |Format| image:: https://img.shields.io/pypi/format/graphviz.svg
    :target: https://pypi.org/project/graphviz/
    :alt: Format

.. |Travis| image:: https://travis-ci.org/xflr6/graphviz.svg?branch=master
    :target: https://travis-ci.org/xflr6/graphviz
    :alt: Travis
.. |Codecov| image:: https://codecov.io/gh/xflr6/graphviz/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/xflr6/graphviz
    :alt: Codecov
.. |Readthedocs-stable| image:: https://readthedocs.org/projects/graphviz/badge/?version=stable
    :target: https://graphviz.readthedocs.io/en/stable/?badge=stable
    :alt: Readthedocs stable
.. |Readthedocs-latest| image:: https://readthedocs.org/projects/graphviz/badge/?version=latest
    :target: https://graphviz.readthedocs.io/en/latest/?badge=latest
    :alt: Readthedocs latest