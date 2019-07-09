#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from library.python import resource

import pytest
import ssl
# import urllib2


class TestRequest(object):
    @pytest.fixture
    def ctx(self):
        r = resource.find("/builtin/cacert")
        # ssl.create_default_context expects unicode string for pem-coded certificates
        r = r.decode('ascii', errors='ignore')
        return ssl.create_default_context(cadata=r)

    def test_certs(self, ctx):
        assert any(
            any(item[0] == ("commonName", "YandexInternalRootCA") for item in cert["subject"])
            for cert in ctx.get_ca_certs()
        )
        assert any(
            any(item[0] == ("commonName", "Certum Trusted Network CA") for item in cert["subject"])
            for cert in ctx.get_ca_certs()
        )

    # def test_internal(self, ctx):
    #     connection = urllib2.urlopen("https://nanny.yandex-team.ru/", context=ctx)
    #     assert connection.read()

    # def test_external(self, ctx):
    #     connection = urllib2.urlopen("https://docs.python.org/", context=ctx)
    #     assert connection.read()
