This include-only library enables simultaneous bindings into Python2 and Python3 with single build.

It provides the following:
- Let dependencies to headers from both Pythons be seen at once during ya make dependency computation. This makes depenency graph more stable.
- Steers build to proper Python headers depending on mode in which binding is built.
- Adds proper Python library to link.

Headers are automatically generated from Python2 and Python3 headers using gen_includes.py script

