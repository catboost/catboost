=============
ABC-Backports
=============

Usage:

.. code-block:: python

    try:
        # ABCs live in "collections.abc" in Python >= 3.3
        from collections.abc import Coroutine, Generator
    except ImportError:
        # fall back to import from "backports_abc"
        from backports_abc import Coroutine, Generator

You can also install the ABCs into the stdlib by calling the ``patch()``
function:

.. code-block:: python

    import backports_abc
    backports_abc.patch()

    try:
        # ABCs live in "collections.abc" in Python >= 3.3
        from collections.abc import Coroutine, Generator
    except ImportError:
        # fall back to import from "collections" in Python <= 3.2
        from backports_abc import Coroutine, Generator

Currently, ``patch()`` provides the following names if missing:

* ``collections.abc.Generator``
* ``collections.abc.Awaitable``
* ``collections.abc.Coroutine``
* ``inspect.isawaitable(obj)``

All of them are also available directly from the ``backports_abc``
module namespace.

In Python 2.x and Python 3.2, it patches the ``collections`` module
instead of the ``collections.abc`` module.  Any names that are already
available when importing this module will not be overwritten.

The names that were previously patched by ``patch()`` can be queried
through the mapping in ``backports_abc.PATCHED``.
