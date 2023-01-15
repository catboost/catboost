RECURSE(
    benchmark
    cython
    fuzzing
    ut
    style
)

IF (OS_LINUX)
    RECURSE(
    sym_versions
)
ENDIF()
