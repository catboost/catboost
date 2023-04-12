=================
Typing Extensions
=================

.. image:: https://badges.gitter.im/python/typing.svg
 :alt: Chat at https://gitter.im/python/typing
 :target: https://gitter.im/python/typing?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Overview
========

The ``typing`` module was added to the standard library in Python 3.5 on
a provisional basis and will no longer be provisional in Python 3.7. However,
this means users of Python 3.5 - 3.6 who are unable to upgrade will not be
able to take advantage of new types added to the ``typing`` module, such as
``typing.Text`` or ``typing.Coroutine``.

The ``typing_extensions`` module contains both backports of these changes
as well as experimental types that will eventually be added to the ``typing``
module, such as ``Protocol`` (see PEP 544 for details about protocols and
static duck typing) or ``TypedDict`` (see PEP 589).

Users of other Python versions should continue to install and use
the ``typing`` module from PyPi instead of using this one unless
specifically writing code that must be compatible with multiple Python
versions or requires experimental types.

Included items
==============

This module currently contains the following:

All Python versions:
--------------------

- ``ClassVar``
- ``ContextManager``
- ``Counter``
- ``DefaultDict``
- ``Deque``
- ``final``
- ``Final``
- ``Literal``
- ``NewType``
- ``NoReturn``
- ``overload`` (note that older versions of ``typing`` only let you use ``overload`` in stubs)
- ``OrderedDict``
- ``Protocol`` (except on Python 3.5.0)
- ``runtime_checkable`` (except on Python 3.5.0)
- ``Text``
- ``Type``
- ``TypedDict``
- ``TypeAlias``
- ``TYPE_CHECKING``

Python 3.4+ only:
-----------------

- ``ChainMap``
- ``ParamSpec``
- ``Concatenate``
- ``ParamSpecArgs``
- ``ParamSpecKwargs``
- ``TypeGuard``

Python 3.5+ only:
-----------------

- ``Annotated`` (except on Python 3.5.0-3.5.2)
- ``AsyncIterable``
- ``AsyncIterator``
- ``AsyncContextManager``
- ``Awaitable``
- ``Coroutine``

Python 3.6+ only:
-----------------

- ``AsyncGenerator``

Other Notes and Limitations
===========================

There are a few types whose interface was modified between different
versions of typing. For example, ``typing.Sequence`` was modified to
subclass ``typing.Reversible`` as of Python 3.5.3.

These changes are _not_ backported to prevent subtle compatibility
issues when mixing the differing implementations of modified classes.

Certain types have incorrect runtime behavior due to limitations of older
versions of the typing module.  For example, ``ParamSpec`` and ``Concatenate``
will not work with ``get_args``, ``get_origin``. Certain PEP 612 special cases
in user-defined ``Generic``\ s are also not available.
These types are only guaranteed to work for static type checking.

Running tests
=============

To run tests, navigate into the appropriate source directory and run
``test_typing_extensions.py``. You will also need to install the latest
version of ``typing`` if you are using a version of Python that does not
include ``typing`` as a part of the standard library.

