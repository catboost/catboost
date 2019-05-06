PY23_LIBRARY()



PEERDIR(
    contrib/python/dateutil
    contrib/python/pytest
    contrib/python/six
    contrib/python/freezegun
    contrib/python/hypothesis
)

TEST_SRCS(
    property/test_isoparse_prop.py
    property/test_parser_prop.py
    __init__.py
    _common.py
    conftest.py
    test_easter.py
    test_import_star.py
    test_imports.py
    test_internals.py
    test_isoparser.py
    test_parser.py
    test_relativedelta.py
    test_rrule.py
    test_tz.py
    test_utils.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
