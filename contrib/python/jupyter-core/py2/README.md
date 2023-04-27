# Jupyter Core

Core common functionality of Jupyter projects.

This package contains base application classes and configuration inherited by other projects.
It doesn't do much on its own.


# Development Setup

The [Jupyter Contributor Guides](http://jupyter.readthedocs.io/en/latest/contributor/content-contributor.html) provide extensive information on contributing code or documentation to Jupyter projects. The limited instructions below for setting up a development environment are for your convenience.

## Coding

You'll need Python and `pip` on the search path. Clone the Jupyter Core git repository to your computer, for example in `/my/projects/jupyter_core`.
Now create an [editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
and download the dependencies of code and test suite by executing:

    cd /my/projects/jupyter_core/
    pip install -e .
    pip install -r dev-requirements.txt
    py.test

The last command runs the test suite to verify the setup. During development, you can pass filenames to `py.test`, and it will execute only those tests.

## Documentation

The documentation of Jupyter Core is generated from the files in `docs/` using Sphinx. Instructions for setting up Sphinx with a selection of optional modules are in the [Documentation Guide](http://jupyter.readthedocs.io/en/latest/contrib_docs/index.html). You'll also need the `make` command.
For a minimal Sphinx installation to process the Jupyter Core docs, execute:

    pip install sphinx

The following commands build the documentation in HTML format and check for broken links:

    cd /my/projects/jupyter_core/docs/
    make html linkcheck

Point your browser to the following URL to access the generated documentation:

_file:///my/projects/jupyter\_core/docs/\_build/html/index.html_

