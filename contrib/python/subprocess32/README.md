subprocess32
------------
[![PyPI version](https://badge.fury.io/py/subprocess32.svg)](https://badge.fury.io/py/subprocess32)
[![POSIX Build Status](https://travis-ci.org/google/python-subprocess32.svg?branch=master)](https://travis-ci.org/google/python-subprocess32)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/53apbb2jk1uslj0m?svg=true
)](https://ci.appveyor.com/project/gpshead/python-subprocess32)

This is a backport of the Python 3 subprocess module for use on Python 2.
This code has not been tested on Windows or other non-POSIX platforms.

subprocess32 includes many important reliability bug fixes relevant on
POSIX platforms.  The most important of which is a C extension module
used internally to handle the code path between fork() and exec().
This module is reliable when an application is using threads.

Refer to the
[Python 3.5 subprocess documentation](https://docs.python.org/3.5/library/subprocess.html)
for usage information.

* Timeout support backported from Python 3.3 is included.
* The run() API from Python 3.5 was backported in subprocess32 3.5.0.
* Otherwise features are frozen at the 3.2 level.

Usage
-----

The recommend pattern for cross platform code is to use the following:

```python
if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess
```

Or if you fully control your POSIX Python 2.7 installation, this can serve
as a replacement for its subprocess module.  Users will thank you by not
filing concurrency bugs.

Got Bugs?
---------

Try to reproduce them on the latest Python 3.x itself and file bug
reports on [bugs.python.org](https://bugs.python.org/).
Add gregory.p.smith to the Nosy list.

If you have reason to believe the issue is specifically with this backport
and not a problem in Python 3 itself, use the github issue tracker.

-- Gregory P. Smith  _greg@krypto.org_
