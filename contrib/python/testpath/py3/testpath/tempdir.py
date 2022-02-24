"""Extra utilities for temporary directories.

NamedFileInTemporaryDirectory and TemporaryWorkingDirectory from IPython, which
uses the 3-clause BSD license.
"""
from __future__ import print_function

import os as _os
from tempfile import TemporaryDirectory


class NamedFileInTemporaryDirectory(object):
    """Open a file named `filename` in a temporary directory.
    
    This context manager is preferred over :class:`tempfile.NamedTemporaryFile`
    when one needs to reopen the file, because on Windows only one handle on a
    file can be open at a time. You can close the returned handle explicitly
    inside the context without deleting the file, and the context manager will
    delete the whole directory when it exits.

    Arguments `mode` and `bufsize` are passed to `open`.
    Rest of the arguments are passed to `TemporaryDirectory`.
    
    Usage example::
    
        with NamedFileInTemporaryDirectory('myfile', 'wb') as f:
            f.write('stuff')
            f.close()
            # You can now pass f.name to things that will re-open the file
    """
    def __init__(self, filename, mode='w+b', bufsize=-1, **kwds):
        self._tmpdir = TemporaryDirectory(**kwds)
        path = _os.path.join(self._tmpdir.name, filename)
        self.file = open(path, mode, bufsize)

    def cleanup(self):
        self.file.close()
        self._tmpdir.cleanup()

    __del__ = cleanup

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.cleanup()


class TemporaryWorkingDirectory(TemporaryDirectory):
    """
    Creates a temporary directory and sets the cwd to that directory.
    Automatically reverts to previous cwd upon cleanup.

    Usage example::

        with TemporaryWorkingDirectory() as tmpdir:
            ...
    """
    def __enter__(self):
        self.old_wd = _os.getcwd()
        _os.chdir(self.name)
        return super(TemporaryWorkingDirectory, self).__enter__()

    def __exit__(self, exc, value, tb):
        _os.chdir(self.old_wd)
        return super(TemporaryWorkingDirectory, self).__exit__(exc, value, tb)

