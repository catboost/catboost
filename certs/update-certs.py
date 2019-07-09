#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import urllib2

COMMON_ROOT_CAS_URL = "https://mkcert.org/generate"
YANDEX_INTERNAL_CAS_URL = "https://crls.yandex.net/allCAs.pem"

def get_text(url):
    return urllib2.urlopen(url).read()

common_root_cas = get_text(COMMON_ROOT_CAS_URL)
yandex_internal_cas = get_text(YANDEX_INTERNAL_CAS_URL)

with open("cacert.pem", "wt") as target:
    target.write(common_root_cas)
    target.write(yandex_internal_cas)

with open("yandex_internal.pem", "wt") as target:
    target.write(yandex_internal_cas)

