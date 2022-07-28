Graphviz
========

|PyPI version| |License| |Supported Python| |Format|

|Build| |Codecov| |Readthedocs-stable| |Readthedocs-latest|

This package facilitates the creation and rendering of graph descriptions in
the DOT_ language of the Graphviz_ graph drawing software (`upstream repo`_)
from Python.

Create a graph object, assemble the graph by adding nodes and edges, and
retrieve its DOT source code string. Save the source code to a file and render
it with the Graphviz installation of your system.

Use the ``view`` option/method to directly inspect the resulting (PDF, PNG,
SVG, etc.) file with its default application. Graphs can also be rendered
and displayed within `Jupyter notebooks`_ (formerly known as
`IPython notebooks`_,
`example <notebook_>`_, `nbviewer <notebook-nbviewer_>`_)
as well as the `Jupyter QtConsole`_.


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

This package runs under Python 3.6+, use pip_ to install:

.. code:: bash

    $ pip install graphviz

To render the generated DOT source code, you also need to install Graphviz_
(`download page <upstream-download_>`_,
`archived versions <upstream-archived_>`_,
`installation procedure for Windows <upstream-windows_>`_).

Make sure that the directory containing the ``dot`` executable is on your
systems' path.

Anaconda_: see the conda-forge_ package
`conda-forge/python-graphviz <conda-forge-python-graphviz_>`_
(`feedstock <conda-forge-python-graphviz-feedstock_>`_),
which should automatically ``conda install``
`conda-forge/graphviz <conda-forge-graphviz_>`_
(`feedstock <conda-forge-graphviz-feedstock_>`_) as dependency.


Quickstart
----------

Create a graph object:

.. code:: python

    >>> import graphviz
    >>> dot = graphviz.Digraph(comment='The Round Table')
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
- graphviz-python_ |--| official Python bindings
  (`documentation <graphviz-python-docs_>`_)
- pydot_ |--| stable pure-Python approach, requires pyparsing


License
-------

This package is distributed under the `MIT license`_.


.. _Graphviz:  https://www.graphviz.org
.. _DOT: https://www.graphviz.org/doc/info/lang.html
.. _upstream repo: https://gitlab.com/graphviz/graphviz/
.. _upstream-download: https://www.graphviz.org/download/
.. _upstream-archived: https://www2.graphviz.org/Archive/stable/
.. _upstream-windows: https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224

.. _pip: https://pip.readthedocs.io

.. _Jupyter notebooks: https://jupyter.org
.. _IPython notebooks: https://ipython.org/notebook.html
.. _Jupyter QtConsole: https://qtconsole.readthedocs.io

.. _notebook: https://github.com/xflr6/graphviz/blob/master/examples/graphviz-notebook.ipynb
.. _notebook-nbviewer: https://nbviewer.jupyter.org/github/xflr6/graphviz/blob/master/examples/graphviz-notebook.ipynb

.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _conda-forge: https://conda-forge.org
.. _conda-forge-python-graphviz: https://anaconda.org/conda-forge/python-graphviz
.. _conda-forge-python-graphviz-feedstock: https://github.com/conda-forge/python-graphviz-feedstock
.. _conda-forge-graphviz: https://anaconda.org/conda-forge/graphviz
.. _conda-forge-graphviz-feedstock: https://github.com/conda-forge/graphviz-feedstock

.. _pygraphviz: https://pypi.org/project/pygraphviz/
.. _graphviz-python: https://pypi.org/project/graphviz-python/
.. _graphviz-python-docs: https://www.graphviz.org/pdf/gv.3python.pdf
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

.. |Build| image:: https://github.com/xflr6/graphviz/actions/workflows/build.yaml/badge.svg?branch=master
    :target: https://github.com/xflr6/graphviz/actions/workflows/build.yaml?query=branch%3Amaster
    :alt: Build
.. |Codecov| image:: https://codecov.io/gh/xflr6/graphviz/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/xflr6/graphviz
    :alt: Codecov
.. |Readthedocs-stable| image:: https://readthedocs.org/projects/graphviz/badge/?version=stable
    :target: https://graphviz.readthedocs.io/en/stable/?badge=stable
    :alt: Readthedocs stable
.. |Readthedocs-latest| image:: https://readthedocs.org/projects/graphviz/badge/?version=latest
    :target: https://graphviz.readthedocs.io/en/latest/?badge=latest
    :alt: Readthedocs latest