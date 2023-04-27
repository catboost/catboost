# Jupyter Client

[![Code Health](https://landscape.io/github/jupyter/jupyter_client/master/landscape.svg?style=flat)](https://landscape.io/github/jupyter/jupyter_client/master)


`jupyter_client` contains the reference implementation of the [Jupyter protocol][].
It also provides client and kernel management APIs for working with kernels.

It also provides the `jupyter kernelspec` entrypoint
for installing kernelspecs for use with Jupyter frontends.

[Jupyter protocol]: https://jupyter-client.readthedocs.io/en/latest/messaging.html


# Development Setup

The [Jupyter Contributor Guides](http://jupyter.readthedocs.io/en/latest/contributor/content-contributor.html) provide extensive information on contributing code or documentation to Jupyter projects. The limited instructions below for setting up a development environment are for your convenience.

## Coding

You'll need Python and `pip` on the search path. Clone the Jupyter Client git repository to your computer, for example in `/my/project/jupyter_client`.
Now create an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
and download the dependencies of code and test suite by executing:

    cd /my/projects/jupyter_client/
    pip install -e .[test]
    py.test

The last command runs the test suite to verify the setup. During development, you can pass filenames to `py.test`, and it will execute only those tests.

## Documentation

The documentation of Jupyter Client is generated from the files in `docs/` using Sphinx. Instructions for setting up Sphinx with a selection of optional modules are in the [Documentation Guide](http://jupyter.readthedocs.io/en/latest/contrib_docs/index.html). You'll also need the `make` command.
For a minimal Sphinx installation to process the Jupyter Client docs, execute:

    pip install ipykernel sphinx sphinx_rtd_theme

The following commands build the documentation in HTML format and check for broken links:

    cd /my/projects/jupyter_client/docs/
    make html linkcheck

Point your browser to the following URL to access the generated documentation:

_file:///my/projects/jupyter\_client/docs/\_build/html/index.html_

