PY3TEST()

LICENSE(BSD)



PEERDIR(
    contrib/python/traitlets
)

SRCDIR(contrib/python/traitlets/py3)

TEST_SRCS(
    traitlets/config/tests/test_application.py
    traitlets/config/tests/test_configurable.py
    traitlets/config/tests/test_loader.py
    traitlets/tests/test_traitlets.py
    traitlets/tests/test_traitlets_enum.py
    traitlets/utils/tests/test_bunch.py
    traitlets/utils/tests/test_decorators.py
    traitlets/utils/tests/test_importstring.py
)

NO_LINT()

END()
