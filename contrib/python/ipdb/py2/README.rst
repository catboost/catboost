IPython `pdb`
=============

.. image:: https://travis-ci.org/gotcha/ipdb.png?branch=master
  :target: https://travis-ci.org/gotcha/ipdb
.. image:: https://codecov.io/gh/gotcha/ipdb/branch/master/graphs/badge.svg?style=flat
  :target: https://codecov.io/gh/gotcha/ipdb?branch=master

Use
---

ipdb exports functions to access the IPython_ debugger, which features
tab completion, syntax highlighting, better tracebacks, better introspection
with the same interface as the `pdb` module.

Example usage:

.. code-block:: python

        import ipdb
        ipdb.set_trace()
        ipdb.set_trace(context=5)  # will show five lines of code
                                   # instead of the default three lines
                                   # or you can set it via IPDB_CONTEXT_SIZE env variable
                                   # or setup.cfg file
        ipdb.pm()
        ipdb.run('x[0] = 3')
        result = ipdb.runcall(function, arg0, arg1, kwarg='foo')
        result = ipdb.runeval('f(1,2) - 3')


Arguments for `set_trace`
+++++++++++++++++++++++++

The `set_trace` function accepts `context` which will show as many lines of code as defined,
and `cond`, which accepts boolean values (such as `abc == 17`) and will start ipdb's
interface whenever `cond` equals to `True`.

Using configuration file
++++++++++++++++++++++++

It's possible to set up context using a `.ipdb` file on your home folder, `setup.cfg`
or `pyproject.toml` on your project folder. You can also set your file location via
env var `$IPDB_CONFIG`. Your environment variable has priority over the home
configuration file, which in turn has priority over the setup config file.
Currently, only context setting is available.

A valid setup.cfg is as follows

::

        [ipdb]
        context=5


A valid .ipdb is as follows

::

        context=5


A valid pyproject.toml is as follows

::

        [tool.ipdb]
        context=5


The post-mortem function, ``ipdb.pm()``, is equivalent to the magic function
``%debug``.

.. _IPython: http://ipython.org

If you install ``ipdb`` with a tool which supports ``setuptools`` entry points,
an ``ipdb`` script is made for you. You can use it to debug your python 2 scripts like

::

        $ bin/ipdb mymodule.py

And for python 3

::

        $ bin/ipdb3 mymodule.py

Alternatively with Python 2.7 only, you can also use

::

        $ python -m ipdb mymodule.py

You can also enclose code with the ``with`` statement to launch ipdb if an exception is raised:

.. code-block:: python

        from ipdb import launch_ipdb_on_exception

        with launch_ipdb_on_exception():
            [...]

.. warning::
   Context managers were introduced in Python 2.5.
   Adding a context manager implies dropping Python 2.4 support.
   Use ``ipdb==0.6`` with 2.4.

Or you can use ``iex`` as a function decorator to launch ipdb if an exception is raised:

.. code-block:: python

        from ipdb import iex

        @iex
        def main():
            [...]

.. warning::
   Using ``from future import print_function`` for Python 3 compat implies dropping Python 2.5 support.
   Use ``ipdb<=0.8`` with 2.5.

Issues with ``stdout``
----------------------

Some tools, like ``nose`` fiddle with ``stdout``.

Until ``ipdb==0.9.4``, we tried to guess when we should also
fiddle with ``stdout`` to support those tools.
However, all strategies tried until 0.9.4 have proven brittle.

If you use ``nose`` or another tool that fiddles with ``stdout``, you should
explicitly ask for ``stdout`` fiddling by using ``ipdb`` like this

.. code-block:: python

        import ipdb
        ipdb.sset_trace()
        ipdb.spm()

        from ipdb import slaunch_ipdb_on_exception
        with slaunch_ipdb_on_exception():
            [...]


Development
-----------

``ipdb`` source code and tracker are at https://github.com/gotcha/ipdb.

Pull requests should take care of updating the changelog ``HISTORY.txt``.

Under the unreleased section, add your changes and your username.

Manual testing
++++++++++++++

To test your changes, make use of ``manual_test.py``. Create a virtual environment,
install IPython and run ``python manual_test.py`` and check if your changes are in effect.
If possible, create automated tests for better behaviour control.

Automated testing
+++++++++++++++++

To run automated tests locally, create a virtual environment, install `coverage`
and run `coverage run setup.py test`.

Third-party support
-------------------

Products.PDBDebugMode
+++++++++++++++++++++

Zope2 Products.PDBDebugMode_ uses ``ipdb``, if available, in place of ``pdb``.

.. _Products.PDBDebugMode: http://pypi.python.org/pypi/Products.PDBDebugMode

iw.debug
++++++++

iw.debug_ allows you to trigger an ``ipdb`` debugger on any published object
of a Zope2 application.

.. _iw.debug: http://pypi.python.org/pypi/iw.debug

ipdbplugin
++++++++++

ipdbplugin_ is a nose_ test runner plugin that also uses the IPython debugger
instead of ``pdb``. (It does not depend on ``ipdb`` anymore).

.. _ipdbplugin: http://pypi.python.org/pypi/ipdbplugin
.. _nose: http://readthedocs.org/docs/nose
