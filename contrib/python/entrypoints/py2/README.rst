Entry points are a way for Python packages to advertise objects with some
common interface. The most common examples are ``console_scripts`` entry points,
which define shell commands by identifying a Python function to run.

*Groups* of entry points, such as ``console_scripts``, point to objects with
similar interfaces. An application might use a group to find its plugins, or
multiple groups if it has different kinds of plugins.

The **entrypoints** module contains functions to find and load entry points.
You can install it from PyPI with ``pip install entrypoints``.

To advertise entry points when distributing a package, see
`entry_points in the Python Packaging User Guide
<https://packaging.python.org/en/latest/distributing.html#entry-points>`_.
