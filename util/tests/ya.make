RECURSE(
    benchmark
    cython
    fuzzing
    style
)

IF (OS_LINUX)
    RECURSE(
    sym_versions
)
ENDIF()
