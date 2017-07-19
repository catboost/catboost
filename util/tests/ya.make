RECURSE(
    benchmark
    cython
    fuzzing
    ut
)

IF (OS_LINUX)
    RECURSE(
    sym_versions
)
ENDIF()
