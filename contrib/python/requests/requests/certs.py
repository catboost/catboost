#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
certs.py
~~~~~~~~

This module returns the preferred default CA certificate bundle.

If you are packaging Requests, e.g., for a Linux distribution or a managed
environment, you can change the definition of where() to return a separately
packaged CA bundle.
"""
import os.path
import ssl

try:
    # load_verify_locations expects PEM cadata to be an ASCII-only unicode object,
    # so we discard unicode in comments.
    builtin_cadata = __loader__.get_data('cacert').decode('ASCII', errors='ignore')
except NameError:
    # Support import from the filesystem for unit2 test runner during elliptics packaging.
    builtin_ca = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cacert.pem'))
else:
    def builtin_ca():
        return None, None, builtin_cadata


def where():
    """Return the preferred certificate bundle."""
    # vendored bundle inside Requests
    return builtin_ca


if hasattr(ssl, 'SSLContext'):
    # Support import from Python older than 2.7.9.
    load_verify_locations = ssl.SSLContext.load_verify_locations

    def load_verify_locations__callable(self, cafile=None, capath=None, cadata=None):
        if callable(cafile):
            cafile, capath, cadata = cafile()

        return load_verify_locations(self, cafile, capath, cadata)

    ssl.SSLContext.load_verify_locations = load_verify_locations__callable
