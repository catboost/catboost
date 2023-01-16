# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

sys.path.insert(0, os.path.abspath(os.environ['CATBOOST_SPARK_MODULE_PATH']))

# -- Project information -----------------------------------------------------

project = 'Catboost for PySpark'
copyright = '2021, CatBoost developers'
author = 'CatBoost developers'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    'numpydoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

def get_logo_path():
    possible_locations = ['../../../../../../../', '../../../../../../github_toplevel']
    for location in possible_locations:
        full_path = os.path.join(location, 'logo', 'catboost.png')
        if os.path.exists(full_path):
            return full_path
    raise Exeception('CatBoost logo path is not found')

html_logo = get_logo_path()

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"]
}

# -- Options for automodapi ----------------------------------------------

numpydoc_show_class_members = False
