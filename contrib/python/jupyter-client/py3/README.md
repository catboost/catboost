# Jupyter Client

[![Build Status](https://github.com/jupyter/jupyter_client/workflows/CI/badge.svg)](https://github.com/jupyter/jupyter_client/actions)
[![Documentation Status](https://readthedocs.org/projects/jupyter-client/badge/?version=latest)](http://jupyter-client.readthedocs.io/en/latest/?badge=latest)

`jupyter_client` contains the reference implementation of the [Jupyter protocol].
It also provides client and kernel management APIs for working with kernels.

It also provides the `jupyter kernelspec` entrypoint
for installing kernelspecs for use with Jupyter frontends.

## Development Setup

The [Jupyter Contributor Guides](https://jupyter.readthedocs.io/en/latest/contributing/content-contributor.html) provide extensive information on contributing code or documentation to Jupyter projects. The limited instructions below for setting up a development environment are for your convenience.

## Coding

You'll need Python and `pip` on the search path. Clone the Jupyter Client git repository to your computer, for example in `/my/project/jupyter_client`

```bash
cd /my/projects/
git clone git@github.com:jupyter/jupyter_client.git
```

Now create an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
and download the dependencies of code and test suite by executing:

```bash
cd /my/projects/jupyter_client/
pip install -e ".[test]"
pytest
```

The last command runs the test suite to verify the setup. During development, you can pass filenames to `pytest`, and it will execute only those tests.

## Documentation

The documentation of Jupyter Client is generated from the files in `docs/` using Sphinx. Instructions for setting up Sphinx with a selection of optional modules are in the [Documentation Guide](https://jupyter.readthedocs.io/en/latest/contributing/docs-contributions/index.html). You'll also need the `make` command.
For a minimal Sphinx installation to process the Jupyter Client docs, execute:

```bash
pip install ".[doc]"
```

The following commands build the documentation in HTML format and check for broken links:

```bash
cd /my/projects/jupyter_client/docs/
make html linkcheck
```

Point your browser to the following URL to access the generated documentation:

_file:///my/projects/jupyter_client/docs/\_build/html/index.html_

## Contributing

`jupyter-client` has adopted automatic code formatting so you shouldn't
need to worry too much about your code style.
As long as your code is valid,
the pre-commit hook should take care of how it should look.
You can invoke the pre-commit hook by hand at any time with:

```bash
pre-commit run
```

which should run any autoformatting on your code
and tell you about any errors it couldn't fix automatically.
You may also install [black integration](https://black.readthedocs.io/en/stable/integrations/editors.html)
into your text editor to format code automatically.

If you have already committed files before setting up the pre-commit
hook with `pre-commit install`, you can fix everything up using
`pre-commit run --all-files`. You need to make the fixing commit
yourself after that.

Some of the hooks only run on CI by default, but you can invoke them by
running with the `--hook-stage manual` argument.

## About the Jupyter Development Team

The Jupyter Development Team is the set of all contributors to the Jupyter project.
This includes all of the Jupyter subprojects.

The core team that coordinates development on GitHub can be found here:
https://github.com/jupyter/.

## Our Copyright Policy

Jupyter uses a shared copyright model. Each contributor maintains copyright
over their contributions to Jupyter. But, it is important to note that these
contributions are typically only changes to the repositories. Thus, the Jupyter
source code, in its entirety is not the copyright of any single person or
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

[jupyter protocol]: https://jupyter-client.readthedocs.io/en/latest/messaging.html
