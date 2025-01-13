.. image:: https://img.shields.io/pypi/v/jaraco.text.svg
   :target: https://pypi.org/project/jaraco.text

.. image:: https://img.shields.io/pypi/pyversions/jaraco.text.svg

.. image:: https://github.com/jaraco/jaraco.text/actions/workflows/main.yml/badge.svg
   :target: https://github.com/jaraco/jaraco.text/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://readthedocs.org/projects/jaracotext/badge/?version=latest
   :target: https://jaracotext.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2024-informational
   :target: https://blog.jaraco.com/skeleton

.. image:: https://tidelift.com/badges/package/pypi/jaraco.text
   :target: https://tidelift.com/subscription/pkg/pypi-jaraco.text?utm_source=pypi-jaraco.text&utm_medium=readme


This package provides handy routines for dealing with text, such as
wrapping, substitution, trimming, stripping, prefix and suffix removal,
line continuation, indentation, comment processing, identifier processing,
values parsing, case insensitive comparison, and more. See the docs
(linked in the badge above) for the detailed documentation and examples.

Layouts
=======

One of the features of this package is the layouts module, which
provides a simple example of translating keystrokes from one keyboard
layout to another::

    echo qwerty | python -m jaraco.text.to-dvorak
    ',.pyf
    echo  "',.pyf" | python -m jaraco.text.to-qwerty
    qwerty

Newline Reporting
=================

Need to know what newlines appear in a file?

::

    $ python -m jaraco.text.show-newlines README.rst
    newline is '\n'

For Enterprise
==============

Available as part of the Tidelift Subscription.

This project and the maintainers of thousands of other packages are working with Tidelift to deliver one enterprise subscription that covers all of the open source you use.

`Learn more <https://tidelift.com/subscription/pkg/pypi-jaraco.text?utm_source=pypi-jaraco.text&utm_medium=referral&utm_campaign=github>`_.
