Welcome to Kiwi
===============

.. image:: https://travis-ci.org/nucleic/kiwi.svg?branch=main
    :target: https://travis-ci.org/nucleic/kiwi
.. image:: https://github.com/nucleic/kiwi/workflows/Continuous%20Integration/badge.svg
    :target: https://github.com/nucleic/kiwi/actions
.. image:: https://github.com/nucleic/kiwi/workflows/Documentation%20building/badge.svg
    :target: https://github.com/nucleic/kiwi/actions
.. image:: https://codecov.io/gh/nucleic/kiwi/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/nucleic/kiwi
.. image:: https://readthedocs.org/projects/kiwisolver/badge/?version=latest
    :target: https://kiwisolver.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Kiwi is an efficient C++ implementation of the Cassowary constraint solving
algorithm. Kiwi is an implementation of the algorithm based on the
`seminal Cassowary paper <https://constraints.cs.washington.edu/solvers/cassowary-tochi.pdf>`_.
It is *not* a refactoring of the original C++ solver. Kiwi has been designed
from the ground up to be lightweight and fast. Kiwi ranges from 10x to 500x
faster than the original Cassowary solver with typical usecases gaining a 40x
improvement. Memory savings are consistently > 5x.

In addition to the C++ solver, Kiwi ships with hand-rolled Python bindings for
Python 3.7+.
