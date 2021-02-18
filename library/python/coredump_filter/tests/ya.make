PY3TEST()



TEST_SRCS(
    test_core_proc.py
    test_core_proc_utils.py
    test_completed_stack.py
)

PEERDIR(
    library/python/coredump_filter
)

DATA(
    arcadia/library/python/coredump_filter/tests
)

END()
