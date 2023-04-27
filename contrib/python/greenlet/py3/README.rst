.. This file is included into docs/history.rst

.. image:: https://github.com/python-greenlet/greenlet/workflows/tests/badge.svg
   :target: https://github.com/python-greenlet/greenlet/actions

Greenlets are lightweight coroutines for in-process concurrent
programming.

The "greenlet" package is a spin-off of `Stackless`_, a version of
CPython that supports micro-threads called "tasklets". Tasklets run
pseudo-concurrently (typically in a single or a few OS-level threads)
and are synchronized with data exchanges on "channels".

A "greenlet", on the other hand, is a still more primitive notion of
micro-thread with no implicit scheduling; coroutines, in other words.
This is useful when you want to control exactly when your code runs.
You can build custom scheduled micro-threads on top of greenlet;
however, it seems that greenlets are useful on their own as a way to
make advanced control flow structures. For example, we can recreate
generators; the difference with Python's own generators is that our
generators can call nested functions and the nested functions can
yield values too. (Additionally, you don't need a "yield" keyword. See
the example in `test_generator.py
<https://github.com/python-greenlet/greenlet/blob/adca19bf1f287b3395896a8f41f3f4fd1797fdc7/src/greenlet/tests/test_generator.py#L1>`_).

Greenlets are provided as a C extension module for the regular unmodified
interpreter.

.. _`Stackless`: http://www.stackless.com


Who is using Greenlet?
======================

There are several libraries that use Greenlet as a more flexible
alternative to Python's built in coroutine support:

 - `Concurrence`_
 - `Eventlet`_
 - `Gevent`_

.. _Concurrence: http://opensource.hyves.org/concurrence/
.. _Eventlet: http://eventlet.net/
.. _Gevent: http://www.gevent.org/

Getting Greenlet
================

The easiest way to get Greenlet is to install it with pip::

  pip install greenlet


Source code archives and binary distributions are available on the
python package index at https://pypi.org/project/greenlet

The source code repository is hosted on github:
https://github.com/python-greenlet/greenlet

Documentation is available on readthedocs.org:
https://greenlet.readthedocs.io
