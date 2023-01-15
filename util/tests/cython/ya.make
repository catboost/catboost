PY23_TEST()


SUBSCRIBER(g:util-subscribers)

SRCDIR(util)

NO_WSHADOW()

PY_SRCS(
    NAMESPACE
    util
    folder/path_ut.pyx
    generic/array_ref_ut.pyx
    generic/deque_ut.pyx
    generic/maybe_ut.pyx
    generic/ptr_ut.pyx
    generic/string_ut.pyx
    generic/vector_ut.pyx
    generic/list_ut.pyx
    generic/hash_set_ut.pyx
    generic/hash_ut.pyx
    memory/blob_ut.pyx
    stream/str_ut.pyx
    string/cast_ut.pyx
    system/types_ut.pyx
    digest/multi_ut.pyx
)

TEST_SRCS(
    test_digest.py
    test_folder.py
    test_generic.py
    test_memory.py
    test_stream.py
    test_system.py
)

END()
