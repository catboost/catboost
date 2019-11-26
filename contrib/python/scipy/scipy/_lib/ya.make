PY23_LIBRARY()



NO_LINT()

PY_SRCS(
    NAMESPACE scipy._lib

    __init__.py
    _gcutils.py
    _numpy_compat.py
#    _testutils.py
    _threadsafety.py
    _tmpdirs.py
    _util.py
    _version.py
    decorator.py
    six.py
)

END()
