PY3TEST()



PEERDIR(
    contrib/python/pytest
    contrib/python/six
    contrib/python/freezegun
    contrib/python/mock
)

TEST_SRCS(
    # too much time to import hypothesis lib
    # property/test_isoparse_prop.py
    # property/test_parser_prop.py

    contrib/python/dateutil/tests/__init__.py
    contrib/python/dateutil/tests/_common.py
    contrib/python/dateutil/tests/test_easter.py
    contrib/python/dateutil/tests/test_import_star.py
    contrib/python/dateutil/tests/test_imports.py
    contrib/python/dateutil/tests/test_internals.py
    contrib/python/dateutil/tests/test_isoparser.py
    contrib/python/dateutil/tests/test_parser.py
    contrib/python/dateutil/tests/test_relativedelta.py
    contrib/python/dateutil/tests/test_rrule.py
    contrib/python/dateutil/tests/test_tz.py
    contrib/python/dateutil/tests/test_utils.py
)

NO_LINT()

END()