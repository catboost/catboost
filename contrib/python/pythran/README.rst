Pythran
#######

https://pythran.readthedocs.io

What is it?
-----------

Pythran is an ahead of time compiler for a subset of the Python language, with a
focus on scientific computing. It takes a Python module annotated with a few
interface descriptions and turns it into a native Python module with the same
interface, but (hopefully) faster.

It is meant to efficiently compile **scientific programs**, and takes advantage
of multi-cores and SIMD instruction units.

Until 0.9.5 (included), Pythran was supporting Python 3 and Python 2.7.
It now only supports Python **3**.

Installation
------------

Pythran sources are hosted on https://github.com/serge-sans-paille/pythran.

Pythran releases are hosted on https://pypi.python.org/pypi/pythran.

Pythran is available on conda-forge on https://anaconda.org/conda-forge/pythran.

Debian/Ubuntu
=============

Using ``pip``
*************

1. Gather dependencies:

   Pythran depends on a few Python modules and several C++ libraries. On a debian-like platform, run::

        $> sudo apt-get install libatlas-base-dev
        $> sudo apt-get install python-dev python-ply python-numpy

2. Install with ``pip``::

        $> pip install pythran

Using ``mamba`` or ``conda``
****************************

1. Using ``mamba`` (https://github.com/conda-forge/miniforge#mambaforge) or ``conda`` (https://github.com/conda-forge/miniforge)
 
2. Run::

       $> mamba install -c conda-forge pythran

   or::

       $> conda install -c conda-forge pythran

Mac OSX
=======

Using brew (https://brew.sh/)::

    $> pip install pythran
    $> brew install openblas
    $> printf '[compiler]\nblas=openblas\ninclude_dirs=/usr/local/opt/openblas/include\nlibrary_dirs=/usr/local/opt/openblas/lib' > ~/.pythranrc

Depending on your setup, you may need to add the following to your ``~/.pythranrc`` file::

    [compiler]
    CXX=g++-4.9
    CC=gcc-4.9

ArchLinux
=========

Using ``pacman``::

    $> pacman -S python-pythran


Fedora
======

Using ``dnf``::

    $> dnf install pythran

Windows
=======

Windows support is on going and only targets Python 3.5+ with either Visual Studio 2017 or, better, clang-cl::

    $> pip install pythran

Note that using ``clang-cl.exe`` is the default setting. It can be changed
through the ``CXX`` and ``CC`` environment variables.


Other Platform
==============

See MANUAL file.


Basic Usage
-----------

A simple pythran input could be ``dprod.py``

.. code-block:: python

    """
    Naive dotproduct! Pythran supports numpy.dot
    """
    #pythran export dprod(int list, int list)
    def dprod(l0,l1):
        """WoW, generator expression, zip and sum."""
        return sum(x * y for x, y in zip(l0, l1))


To turn it into a native module, run::

    $> pythran dprod.py

That will generate a native dprod.so that can be imported just like the former
module::

    $> python -c 'import dprod' # this imports the native module instead


Documentation
-------------

The user documentation is available in the MANUAL file from the doc directory.

The developer documentation is available in the DEVGUIDE file from the doc
directory. There is also a TUTORIAL file for those who don't like reading
documentation.

The CLI documentation is available from the pythran help command::

    $> pythran --help

Some extra developer documentation is also available using pydoc. Beware, this
is the computer science incarnation for the famous Where's Waldo? game::

    $> pydoc pythran
    $> pydoc pythran.typing
    $> pydoc -b  # in the browser


Examples
--------

See the ``pythran/tests/cases/`` directory from the sources.


Contact
-------

Praise, flame and cookies:

- pythran@freelists.org -- register at https://www.freelists.org/list/pythran first!

- #pythran on OFTC, https://oftc.net 

- serge.guelton@telecom-bretagne.eu

The mailing list archive is available at https://www.freelists.org/archive/pythran/.

Citing
------

If you need to cite a Pythran paper, feel free to use

.. code-block:: bibtex

    @article{guelton2015pythran,
      title={Pythran: Enabling static optimization of scientific python programs},
      author={Guelton, Serge and Brunet, Pierrick and Amini, Mehdi and Merlini,
                      Adrien and Corbillon, Xavier and Raynaud, Alan},
      journal={Computational Science \& Discovery},
      volume={8},
      number={1},
      pages={014001},
      year={2015},
      publisher={IOP Publishing}
    }


Authors
-------

See AUTHORS file.

License
-------

See LICENSE file.
