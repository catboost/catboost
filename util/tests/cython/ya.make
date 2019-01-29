PYTEST()



SRCDIR(util)

NO_WSHADOW()

PY_SRCS(
    NAMESPACE
    util
    generic/array_ref_ut.pyx
    generic/maybe_ut.pyx
    generic/ptr_ut.pyx
    generic/string_ut.pyx
    generic/vector_ut.pyx
    generic/hash_ut.pyx
    string/cast_ut.pyx
    system/types_ut.pyx
    digest/multi_ut.pyx
)

TEST_SRCS(
    test_generic.py
    test_system.py
    test_digest.py
)

END()
