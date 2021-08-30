PY3TEST()



PEERDIR(
    contrib/python/traitlets
)

SRCDIR(contrib/python/traitlets/py3/traitlets)

TEST_SRCS(
    config/tests/test_application.py
    config/tests/test_configurable.py
    config/tests/test_loader.py
    tests/test_traitlets.py
    tests/test_traitlets_enum.py
    utils/tests/test_bunch.py
    utils/tests/test_decorators.py
    utils/tests/test_importstring.py
)

NO_LINT()

END()
