#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import urllib2


SOURCES = [
    "https://mkcert.org/generate",  # Common root CAs
    "https://crls.yandex.net/allCAs.pem",  # YandexInternalRootCA
]

combined = ""
for link in SOURCES:
    connection = urllib2.urlopen (link)
    combined += connection.read ()

with open ("cacert.pem", "wt") as target:
    target.write (combined)

