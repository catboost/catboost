PY23_TEST()



PEERDIR(
    contrib/python/python-dateutil
    contrib/python/freezegun
    contrib/python/hypothesis
)

ENV(LC_ALL=ru_RU.UTF-8)
ENV(LANG=ru_RU.UTF-8)
# because we cannot change TZ in arcadia CI
ENV(DATEUTIL_MAY_NOT_CHANGE_TZ_VAR=1)

SRCDIR(contrib/python/python-dateutil/dateutil/test)

TEST_SRCS(
    property/test_isoparse_prop.py
    property/test_parser_prop.py
    # property/test_tz_prop.py
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
