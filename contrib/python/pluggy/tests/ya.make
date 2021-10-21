PY23_TEST()



PEERDIR (
    contrib/python/pluggy
)

TEST_SRCS(
    conftest.py
    test_deprecations.py
    test_details.py
    test_helpers.py
    test_hookcaller.py
    test_invocations.py
    test_multicall.py
    test_pluginmanager.py
    test_tracer.py
)

NO_LINT()

END()
