###################################################################
Jedi - an awesome autocompletion/static analysis library for Python
###################################################################

.. image:: https://img.shields.io/pypi/v/jedi.svg?style=flat
    :target: https://pypi.python.org/pypi/jedi
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/jedi.svg
    :target: https://pypi.python.org/pypi/jedi
    :alt: Supported Python versions

.. image:: https://travis-ci.org/davidhalter/jedi.svg?branch=master
    :target: https://travis-ci.org/davidhalter/jedi
    :alt: Linux Tests

.. image:: https://ci.appveyor.com/api/projects/status/mgva3bbawyma1new/branch/master?svg=true
    :target: https://ci.appveyor.com/project/davidhalter/jedi/branch/master
    :alt: Windows Tests

.. image:: https://coveralls.io/repos/davidhalter/jedi/badge.svg?branch=master
    :target: https://coveralls.io/r/davidhalter/jedi
    :alt: Coverage status


*If you have specific questions, please add an issue or ask on* `Stack Overflow
<https://stackoverflow.com/questions/tagged/python-jedi>`_ *with the label* ``python-jedi``.


Jedi is a static analysis tool for Python that can be used in IDEs/editors. Its
historic focus is autocompletion, but does static analysis for now as well.
Jedi is fast and is very well tested. It understands Python on a deeper level
than all other static analysis frameworks for Python.

Jedi has support for two different goto functions. It's possible to search for
related names and to list all names in a Python file and infer them. Jedi
understands docstrings and you can use Jedi autocompletion in your REPL as
well.

Jedi uses a very simple API to connect with IDEs. There's a reference
implementation as a `VIM-Plugin <https://github.com/davidhalter/jedi-vim>`_,
which uses Jedi's autocompletion.  We encourage you to use Jedi in your IDEs.
It's really easy.

Jedi can currently be used with the following editors/projects:

- Vim (jedi-vim_, YouCompleteMe_, deoplete-jedi_, completor.vim_)
- Emacs (Jedi.el_, company-mode_, elpy_, anaconda-mode_, ycmd_)
- Sublime Text (SublimeJEDI_ [ST2 + ST3], anaconda_ [only ST3])
- TextMate_ (Not sure if it's actually working)
- Kate_ version 4.13+ supports it natively, you have to enable it, though. [`proof
  <https://projects.kde.org/projects/kde/applications/kate/repository/show?rev=KDE%2F4.13>`_]
- Atom_ (autocomplete-python-jedi_)
- `GNOME Builder`_ (with support for GObject Introspection)
- `Visual Studio Code`_ (via `Python Extension <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_)
- Gedit (gedi_)
- wdb_ - Web Debugger
- `Eric IDE`_ (Available as a plugin)
- `IPython 6.0.0+ <https://ipython.readthedocs.io/en/stable/whatsnew/version6.html>`_

and many more!


Here are some pictures taken from jedi-vim_:

.. image:: https://github.com/davidhalter/jedi/raw/master/docs/_screenshots/screenshot_complete.png

Completion for almost anything (Ctrl+Space).

.. image:: https://github.com/davidhalter/jedi/raw/master/docs/_screenshots/screenshot_function.png

Display of function/class bodies, docstrings.

.. image:: https://github.com/davidhalter/jedi/raw/master/docs/_screenshots/screenshot_pydoc.png

Pydoc support (Shift+k).

There is also support for goto and renaming.

Get the latest version from `github <https://github.com/davidhalter/jedi>`_
(master branch should always be kind of stable/working).

Docs are available at `https://jedi.readthedocs.org/en/latest/
<https://jedi.readthedocs.org/en/latest/>`_. Pull requests with documentation
enhancements and/or fixes are awesome and most welcome. Jedi uses `semantic
versioning <https://semver.org/>`_.

If you want to stay up-to-date (News / RFCs), please subscribe to this `github
thread <https://github.com/davidhalter/jedi/issues/1063>`_.:



Installation
============

    pip install jedi

Note: This just installs the Jedi library, not the editor plugins. For
information about how to make it work with your editor, refer to the
corresponding documentation.

You don't want to use ``pip``? Please refer to the `manual
<https://jedi.readthedocs.org/en/latest/docs/installation.html>`_.


Feature Support and Caveats
===========================

Jedi really understands your Python code. For a comprehensive list what Jedi
understands, see: `Features
<https://jedi.readthedocs.org/en/latest/docs/features.html>`_. A list of
caveats can be found on the same page.

You can run Jedi on CPython 2.7 or 3.4+ but it should also
understand/parse code older than those versions. Additionally you should be able
to use `Virtualenvs <https://jedi.readthedocs.org/en/latest/docs/api.html#environments>`_
very well.

Tips on how to use Jedi efficiently can be found `here
<https://jedi.readthedocs.org/en/latest/docs/features.html#recipes>`_.

API
---

You can find the documentation for the `API here <https://jedi.readthedocs.org/en/latest/docs/api.html>`_.


Autocompletion / Goto / Pydoc
-----------------------------

Please check the API for a good explanation. There are the following commands:

- ``jedi.Script.goto_assignments``
- ``jedi.Script.completions``
- ``jedi.Script.usages``

The returned objects are very powerful and really all you might need.


Autocompletion in your REPL (IPython, etc.)
-------------------------------------------

Starting with IPython `6.0.0` Jedi is a dependency of IPython. Autocompletion
in IPython is therefore possible without additional configuration.

It's possible to have Jedi autocompletion in REPL modes - `example video <https://vimeo.com/122332037>`_.
This means that in Python you can enable tab completion in a `REPL
<https://jedi.readthedocs.org/en/latest/docs/usage.html#tab-completion-in-the-python-shell>`_.


Static Analysis / Linter
------------------------

To do all forms of static analysis, please try to use ``jedi.names``. It will
return a list of names that you can use to infer types and so on.

Linting is another thing that is going to be part of Jedi. For now you can try
an alpha version ``python -m jedi linter``. The API might change though and
it's still buggy. It's Jedi's goal to be smarter than classic linter and
understand ``AttributeError`` and other code issues.


Refactoring
-----------

Jedi's parser would support refactoring, but there's no API to use it right
now.  If you're interested in helping out here, let me know. With the latest
parser changes, it should be very easy to actually make it work.


Development
===========

There's a pretty good and extensive `development documentation
<https://jedi.readthedocs.org/en/latest/docs/development.html>`_.


Testing
=======

The test suite depends on ``tox`` and ``pytest``::

    pip install tox pytest

To run the tests for all supported Python versions::

    tox

If you want to test only a specific Python version (e.g. Python 2.7), it's as
easy as ::

    tox -e py27

Tests are also run automatically on `Travis CI
<https://travis-ci.org/davidhalter/jedi/>`_.

For more detailed information visit the `testing documentation
<https://jedi.readthedocs.org/en/latest/docs/testing.html>`_.


Acknowledgements
================

- Takafumi Arakaki (@tkf) for creating a solid test environment and a lot of
  other things.
- Danilo Bargen (@dbrgn) for general housekeeping and being a good friend :).
- Guido van Rossum (@gvanrossum) for creating the parser generator pgen2
  (originally used in lib2to3).



.. _jedi-vim: https://github.com/davidhalter/jedi-vim
.. _youcompleteme: https://valloric.github.io/YouCompleteMe/
.. _deoplete-jedi: https://github.com/zchee/deoplete-jedi
.. _completor.vim: https://github.com/maralla/completor.vim
.. _Jedi.el: https://github.com/tkf/emacs-jedi
.. _company-mode: https://github.com/syohex/emacs-company-jedi
.. _elpy: https://github.com/jorgenschaefer/elpy
.. _anaconda-mode: https://github.com/proofit404/anaconda-mode
.. _ycmd: https://github.com/abingham/emacs-ycmd
.. _sublimejedi: https://github.com/srusskih/SublimeJEDI
.. _anaconda: https://github.com/DamnWidget/anaconda
.. _wdb: https://github.com/Kozea/wdb
.. _TextMate: https://github.com/lawrenceakka/python-jedi.tmbundle
.. _Kate: https://kate-editor.org
.. _Atom: https://atom.io/
.. _autocomplete-python-jedi: https://atom.io/packages/autocomplete-python-jedi
.. _GNOME Builder: https://wiki.gnome.org/Apps/Builder
.. _Visual Studio Code: https://code.visualstudio.com/
.. _gedi: https://github.com/isamert/gedi
.. _Eric IDE: https://eric-ide.python-projects.org
