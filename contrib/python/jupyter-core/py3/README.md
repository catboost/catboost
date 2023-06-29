# Jupyter Core

[![Build Status](https://github.com/jupyter/jupyter_core/actions/workflows/test.yml/badge.svg?query=branch%3Amain++)](https://github.com/jupyter/jupyter_core/actions/workflows/test.yml/badge.svg?query=branch%3Amain++)
[![Documentation Status](https://readthedocs.org/projects/jupyter-core/badge/?version=latest)](http://jupyter-core.readthedocs.io/en/latest/?badge=latest)

Core common functionality of Jupyter projects.

This package contains base application classes and configuration inherited by other projects.
It doesn't do much on its own.

# Development Setup

The [Jupyter Contributor Guides](https://docs.jupyter.org/en/latest/contributing/content-contributor.html) provide extensive information on contributing code or documentation to Jupyter projects. The limited instructions below for setting up a development environment are for your convenience.

## Coding

You'll need Python and `pip` on the search path. Clone the Jupyter Core git repository to your computer, for example in `/my/projects/jupyter_core`.
Now create an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
and download the dependencies of code and test suite by executing:

```
cd /my/projects/jupyter_core/
pip install -e ".[test]"
py.test
```

The last command runs the test suite to verify the setup. During development, you can pass filenames to `py.test`, and it will execute only those tests.

## Code Styling

`jupyter_core` has adopted automatic code formatting so you shouldn't
need to worry too much about your code style.
As long as your code is valid,
the pre-commit hook should take care of how it should look.
`pre-commit` and its associated hooks will automatically be installed when
you run `pip install -e ".[test]"`

To install `pre-commit` manually, run the following:

```bash
    pip install pre-commit
    pre-commit install
```

You can invoke the pre-commit hook by hand at any time with:

```bash
    pre-commit run
```

which should run any autoformatting on your code
and tell you about any errors it couldn't fix automatically.
You may also install [black integration](https://github.com/psf/black#editor-integration)
into your text editor to format code automatically.

If you have already committed files before setting up the pre-commit
hook with `pre-commit install`, you can fix everything up using
`pre-commit run --all-files`. You need to make the fixing commit
yourself after that.

## Documentation

The documentation of Jupyter Core is generated from the files in `docs/` using Sphinx. Instructions for setting up Sphinx with a selection of optional modules are in the [Documentation Guide](https://docs.jupyter.org/en/latest/contributing/content-contributor.html). You'll also need the `make` command.
For a minimal Sphinx installation to process the Jupyter Core docs, execute:

```
pip install sphinx
```

The following commands build the documentation in HTML format and check for broken links:

```
cd /my/projects/jupyter_core/docs/
make html linkcheck
```

Point your browser to the following URL to access the generated documentation:

_file:///my/projects/jupyter_core/docs/\_build/html/index.html_

## About the Jupyter Development Team

The Jupyter Development Team is the set of all contributors to the Jupyter
project. This includes all of the Jupyter subprojects. A full list with
details is kept in the documentation directory, in the file
`about/credits.txt`.

The core team that coordinates development on GitHub can be found here:
https://github.com/ipython/.

## Our Copyright Policy

Jupyter uses a shared copyright model. Each contributor maintains copyright
over their contributions to Jupyter. It is important to note that these
contributions are typically only changes to the repositories. Thus, the Jupyter
source code in its entirety is not the copyright of any single person or
institution. Instead, it is the collective copyright of the entire Jupyter
Development Team. If individual contributors want to maintain a record of what
changes/contributions they have specific copyright on, they should indicate
their copyright in the commit message of the change, when they commit the
change to one of the Jupyter repositories.

With this in mind, the following banner should be used in any source code file
to indicate the copyright and license terms:

```
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
```
