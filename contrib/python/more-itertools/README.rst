==============
More Itertools
==============

.. image:: https://coveralls.io/repos/github/erikrose/more-itertools/badge.svg?branch=master
  :target: https://coveralls.io/github/erikrose/more-itertools?branch=master

Python's ``itertools`` library is a gem - you can compose elegant solutions
for a variety of problems with the functions it provides. In ``more-itertools``
we collect additional building blocks, recipes, and routines for working with
Python iterables.

----

+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Grouping               | `chunked <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.chunked>`_,                                                                                                                        |
|                        | `sliced <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sliced>`_,                                                                                                                          |
|                        | `distribute <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute>`_,                                                                                                                  |
|                        | `divide <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.divide>`_,                                                                                                                          |
|                        | `split_at <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_at>`_,                                                                                                                      |
|                        | `split_before <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_before>`_,                                                                                                              |
|                        | `split_after <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_after>`_,                                                                                                                |
|                        | `bucket <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.bucket>`_,                                                                                                                          |
|                        | `grouper <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.grouper>`_,                                                                                                                        |
|                        | `partition <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition>`_                                                                                                                     |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Lookahead and lookback | `spy <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.spy>`_,                                                                                                                                |
|                        | `peekable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.peekable>`_,                                                                                                                      |
|                        | `seekable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.seekable>`_                                                                                                                       |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Windowing              | `windowed <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.windowed>`_,                                                                                                                      |
|                        | `stagger <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.stagger>`_,                                                                                                                        |
|                        | `pairwise <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.pairwise>`_                                                                                                                       |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Augmenting             | `count_cycle <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.count_cycle>`_,                                                                                                                |
|                        | `intersperse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.intersperse>`_,                                                                                                                |
|                        | `padded <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padded>`_,                                                                                                                          |
|                        | `adjacent <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.adjacent>`_,                                                                                                                      |
|                        | `groupby_transform <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.groupby_transform>`_,                                                                                                    |
|                        | `padnone <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.padnone>`_,                                                                                                                        |
|                        | `ncycles <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ncycles>`_                                                                                                                         |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Combining              | `collapse <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collapse>`_,                                                                                                                      |
|                        | `sort_together <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.sort_together>`_,                                                                                                            |
|                        | `interleave <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave>`_,                                                                                                                  |
|                        | `interleave_longest <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.interleave_longest>`_,                                                                                                  |
|                        | `collate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.collate>`_,                                                                                                                        |
|                        | `zip_offset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.zip_offset>`_,                                                                                                                  |
|                        | `dotproduct <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.dotproduct>`_,                                                                                                                  |
|                        | `flatten <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.flatten>`_,                                                                                                                        |
|                        | `roundrobin <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.roundrobin>`_,                                                                                                                  |
|                        | `prepend <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.prepend>`_                                                                                                                         |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Summarizing            | `ilen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.ilen>`_,                                                                                                                              |
|                        | `first <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.first>`_,                                                                                                                            |
|                        | `last <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.last>`_,                                                                                                                              |
|                        | `one <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.one>`_,                                                                                                                                |
|                        | `unique_to_each <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_to_each>`_,                                                                                                          |
|                        | `locate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.locate>`_,                                                                                                                          |
|                        | `rlocate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.rlocate>`_,                                                                                                                        |
|                        | `consecutive_groups <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consecutive_groups>`_,                                                                                                  |
|                        | `exactly_n <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.exactly_n>`_,                                                                                                                    |
|                        | `run_length <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.run_length>`_,                                                                                                                  |
|                        | `map_reduce <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.map_reduce>`_,                                                                                                                  |
|                        | `all_equal <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_equal>`_,                                                                                                                    |
|                        | `first_true <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.first_true>`_,                                                                                                                  |
|                        | `nth <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.nth>`_,                                                                                                                                |
|                        | `quantify <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.quantify>`_                                                                                                                       |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Selecting              | `islice_extended <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.islice_extended>`_,                                                                                                        |
|                        | `strip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.strip>`_,                                                                                                                            |
|                        | `lstrip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.lstrip>`_,                                                                                                                          |
|                        | `rstrip <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.rstrip>`_,                                                                                                                          |
|                        | `take <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.take>`_,                                                                                                                              |
|                        | `tail <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.tail>`_,                                                                                                                              |
|                        | `unique_everseen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertoo ls.unique_everseen>`_,                                                                                                       |
|                        | `unique_justseen <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_justseen>`_                                                                                                         |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Combinatorics          | `distinct_permutations <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_permutations>`_,                                                                                            |
|                        | `circular_shifts <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.circular_shifts>`_,                                                                                                        |
|                        | `powerset <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.powerset>`_,                                                                                                                      |
|                        | `random_product <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_product>`_,                                                                                                          |
|                        | `random_permutation <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_permutation>`_,                                                                                                  |
|                        | `random_combination <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_combination>`_,                                                                                                  |
|                        | `random_combination_with_replacement <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.random_combination_with_replacement>`_,                                                                |
|                        | `nth_combination <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.nth_combination>`_                                                                                                         |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Wrapping               | `always_iterable <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.always_iterable>`_,                                                                                                        |
|                        | `consumer <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consumer>`_,                                                                                                                      |
|                        | `with_iter <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.with_iter>`_,                                                                                                                    |
|                        | `iter_except <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.iter_except>`_                                                                                                                 |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Others                 | `replace <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.replace>`_,                                                                                                                        |
|                        | `numeric_range <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.numeric_range>`_,                                                                                                            |
|                        | `always_reversible <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.always_reversible>`_,                                                                                                    |
|                        | `side_effect <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.side_effect>`_,                                                                                                                |
|                        | `iterate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.iterate>`_,                                                                                                                        |
|                        | `difference <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.difference>`_,                                                                                                                  |
|                        | `make_decorator <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.make_decorator>`_,                                                                                                          |
|                        | `SequenceView <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.SequenceView>`_,                                                                                                              |
|                        | `consume <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.consume>`_,                                                                                                                        |
|                        | `accumulate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.accumulate>`_,                                                                                                                  |
|                        | `tabulate <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.tabulate>`_,                                                                                                                      |
|                        | `repeatfunc <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.repeatfunc>`_                                                                                                                   |
+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Getting started
===============

To get started, install the library with `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: shell

    pip install more-itertools

The recipes from the `itertools docs <https://docs.python.org/3/library/itertools.html#itertools-recipes>`_
are included in the top-level package:

.. code-block:: python

    >>> from more_itertools import flatten
    >>> iterable = [(0, 1), (2, 3)]
    >>> list(flatten(iterable))
    [0, 1, 2, 3]

Several new recipes are available as well:

.. code-block:: python

    >>> from more_itertools import chunked
    >>> iterable = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> list(chunked(iterable, 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    >>> from more_itertools import spy
    >>> iterable = (x * x for x in range(1, 6))
    >>> head, iterable = spy(iterable, n=3)
    >>> list(head)
    [1, 4, 9]
    >>> list(iterable)
    [1, 4, 9, 16, 25]



For the full listing of functions, see the `API documentation <https://more-itertools.readthedocs.io/en/latest/api.html>`_.

Development
===========

``more-itertools`` is maintained by `@erikrose <https://github.com/erikrose>`_
and `@bbayles <https://github.com/bbayles>`_, with help from `many others <https://github.com/erikrose/more-itertools/graphs/contributors>`_.
If you have a problem or suggestion, please file a bug or pull request in this
repository. Thanks for contributing!
