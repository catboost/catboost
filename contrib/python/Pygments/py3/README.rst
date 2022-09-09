Welcome to Pygments
===================

This is the source of Pygments.  It is a **generic syntax highlighter** written
in Python that supports over 500 languages and text formats, for use in code
hosting, forums, wikis or other applications that need to prettify source code.

Installing
----------

... works as usual, use ``pip install Pygments`` to get published versions,
or ``python setup.py install`` to install from a checkout.

Documentation
-------------

... can be found online at https://pygments.org/ or created with Sphinx by ::

   make docs

By default, the documentation does not include the demo page, as it requires
having Docker installed for building Pyodide. To build the documentation with
the demo page, use ::

   WEBSITE_BUILD=1 make docs

The initial build might take some time, but subsequent ones should be instant
because of Docker caching.

To view the generated documentation, serve it using Python's ``http.server``
module (this step is required for the demo to work) ::

   python3 -m http.server --directory doc/_build/html


Development
-----------

... takes place on `GitHub <https://github.com/pygments/pygments>`_, where the
Git repository, tickets and pull requests can be viewed.

Continuous testing runs on GitHub workflows:

.. image:: https://github.com/pygments/pygments/workflows/Pygments/badge.svg
   :target: https://github.com/pygments/pygments/actions?query=workflow%3APygments

Contribution guidelines are found in Contributing.md_.

.. _Contributing.md: https://github.com/pygments/pygments/blob/master/Contributing.md

The authors
-----------

Pygments is maintained by **Georg Brandl**, e-mail address *georg*\ *@*\ *python.org*
and **Matthäus Chajdas**.

Many lexers and fixes have been contributed by **Armin Ronacher**, the rest of
the `Pocoo <https://www.pocoo.org/>`_ team and **Tim Hatch**.

The code is distributed under the BSD 2-clause license.  Contributors making pull
requests must agree that they are able and willing to put their contributions
under that license.
