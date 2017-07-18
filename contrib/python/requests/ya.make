LIBRARY()



RESOURCE(
    .dist-info/METADATA      /fs/contrib/python/requests/.dist-info/METADATA
    .dist-info/top_level.txt /fs/contrib/python/requests/.dist-info/top_level.txt
)

PY_SRCS(
    TOP_LEVEL
    requests/adapters.py
    requests/status_codes.py
    requests/api.py
    requests/compat.py
    requests/auth.py
    requests/__init__.py
    requests/utils.py
    requests/certs.py
    requests/hooks.py
    requests/structures.py
    requests/exceptions.py
    requests/cookies.py
    requests/models.py
    requests/sessions.py
    requests/packages/__init__.py
    requests/packages/urllib3/poolmanager.py
    requests/packages/urllib3/_collections.py
    requests/packages/urllib3/__init__.py
    requests/packages/urllib3/connection.py
    requests/packages/urllib3/request.py
    requests/packages/urllib3/fields.py
    requests/packages/urllib3/filepost.py
    requests/packages/urllib3/connectionpool.py
    requests/packages/urllib3/response.py
    requests/packages/urllib3/exceptions.py
    requests/packages/urllib3/packages/ordered_dict.py
    requests/packages/urllib3/packages/__init__.py
    requests/packages/urllib3/packages/six.py
    requests/packages/urllib3/packages/ssl_match_hostname/_implementation.py
    requests/packages/urllib3/packages/ssl_match_hostname/__init__.py
    requests/packages/urllib3/util/timeout.py
    requests/packages/urllib3/util/retry.py
    requests/packages/urllib3/util/ssl_.py
    requests/packages/urllib3/util/__init__.py
    requests/packages/urllib3/util/connection.py
    requests/packages/urllib3/util/url.py
    requests/packages/urllib3/util/request.py
    requests/packages/urllib3/util/response.py
    requests/packages/urllib3/contrib/__init__.py
    requests/packages/urllib3/contrib/pyopenssl.py
    requests/packages/urllib3/contrib/ntlmpool.py
    requests/packages/chardet/langbulgarianmodel.py
    requests/packages/chardet/gb2312freq.py
    requests/packages/chardet/eucjpprober.py
    requests/packages/chardet/euckrfreq.py
    requests/packages/chardet/jpcntx.py
    requests/packages/chardet/langhungarianmodel.py
    requests/packages/chardet/langgreekmodel.py
    requests/packages/chardet/compat.py
    requests/packages/chardet/euckrprober.py
    requests/packages/chardet/langcyrillicmodel.py
    requests/packages/chardet/codingstatemachine.py
    requests/packages/chardet/big5freq.py
    requests/packages/chardet/universaldetector.py
    requests/packages/chardet/euctwfreq.py
    requests/packages/chardet/langthaimodel.py
    requests/packages/chardet/__init__.py
    requests/packages/chardet/chardetect.py
    requests/packages/chardet/euctwprober.py
    requests/packages/chardet/cp949prober.py
    requests/packages/chardet/big5prober.py
    requests/packages/chardet/langhebrewmodel.py
    requests/packages/chardet/utf8prober.py
    requests/packages/chardet/escprober.py
    requests/packages/chardet/charsetgroupprober.py
    requests/packages/chardet/mbcssm.py
    requests/packages/chardet/latin1prober.py
    requests/packages/chardet/sbcsgroupprober.py
    requests/packages/chardet/escsm.py
    requests/packages/chardet/constants.py
    requests/packages/chardet/sbcharsetprober.py
    requests/packages/chardet/hebrewprober.py
    requests/packages/chardet/gb2312prober.py
    requests/packages/chardet/charsetprober.py
    requests/packages/chardet/chardistribution.py
    requests/packages/chardet/mbcharsetprober.py
    requests/packages/chardet/sjisprober.py
    requests/packages/chardet/mbcsgroupprober.py
    requests/packages/chardet/jisfreq.py
)

RESOURCE(
    requests/cacert.pem /py_modules/cacert
)


NO_LINT()

END()
