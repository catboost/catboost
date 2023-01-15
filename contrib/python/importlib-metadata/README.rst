=========================
 ``importlib_metadata``
=========================

``importlib_metadata`` is a library to access the metadata for a Python
package.  It is intended to be ported to Python 3.8.


Usage
=====

See the `online documentation <https://importlib_metadata.readthedocs.io/>`_
for usage details.

`Finder authors
<https://docs.python.org/3/reference/import.html#finders-and-loaders>`_ can
also add support for custom package installers.  See the above documentation
for details.


Caveats
=======

This project primarily supports third-party packages installed by PyPA
tools (or other conforming packages). It does not support:

- Packages in the stdlib.
- Packages installed without metadata.

Project details
===============

 * Project home: https://gitlab.com/python-devs/importlib_metadata
 * Report bugs at: https://gitlab.com/python-devs/importlib_metadata/issues
 * Code hosting: https://gitlab.com/python-devs/importlib_metadata.git
 * Documentation: http://importlib_metadata.readthedocs.io/
