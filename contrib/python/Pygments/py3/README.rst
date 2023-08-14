Welcome to Pygments
===================

This is the source of Pygments.  It is a **generic syntax highlighter** written
in Python that supports over 500 languages and text formats, for use in code
hosting, forums, wikis or other applications that need to prettify source code.

Installing
----------

... works as usual, use ``pip install Pygments`` to get published versions,
or ``pip install -e .`` to install from a checkout in editable mode.

Documentation
-------------

... can be found online at https://pygments.org/ or created with Sphinx by ::

   tox -e doc

By default, the documentation does not include the demo page, as it requires
having Docker installed for building Pyodide. To build the documentation with
the demo page, use ::

   tox -e web-doc

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

Please read our `Contributing instructions <https://pygments.org/docs/contributing>`_.

Security considerations
-----------------------

Pygments provides no guarantees on execution time, which needs to be taken
into consideration when using Pygments to process arbitrary user inputs. For
example, if you have a web service which uses Pygments for highlighting, there
may be inputs which will cause the Pygments process to run "forever" and/or use
significant amounts of memory. This can subsequently be used to perform a
remote denial-of-service attack on the server if the processes are not
terminated quickly.

Unfortunately, it's practically impossible to harden Pygments itself against
those issues: Some regular expressions can result in "catastrophic
backtracking", but other bugs like incorrect matchers can also
cause similar problems, and there is no way to find them in an automated fashion
(short of solving the halting problem.) Pygments has extensive unit tests,
automated randomized testing, and is also tested by `OSS-Fuzz <https://github.com/google/oss-fuzz/tree/master/projects/pygments>`_,
but we will never be able to eliminate all bugs in this area.

Our recommendations are:

* Ensure that the Pygments process is *terminated* after a reasonably short
  timeout. In general Pygments should take seconds at most for reasonably-sized
  input.
* *Limit* the number of concurrent Pygments processes to avoid oversubscription
  of resources.

The Pygments authors will treat any bug resulting in long processing times with
high priority -- it's one of those things that will be fixed in a patch release.
When reporting a bug where you suspect super-linear execution times, please make
sure to attach an input to reproduce it.

The authors
-----------

Pygments is maintained by **Georg Brandl**, e-mail address *georg*\ *@*\ *python.org*, **Matthäus Chajdas** and **Jean Abou-Samra**.

Many lexers and fixes have been contributed by **Armin Ronacher**, the rest of
the `Pocoo <https://www.pocoo.org/>`_ team and **Tim Hatch**.

The code is distributed under the BSD 2-clause license.  Contributors making pull
requests must agree that they are able and willing to put their contributions
under that license.
