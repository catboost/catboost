LIBRARY()

LICENSE(
    BSD
)

VERSION(1.0.5)



NO_COMPILER_WARNINGS()

IF (SANITIZER_TYPE STREQUAL undefined)
    NO_SANITIZE()
ENDIF ()

SRCS(
    snappy.cc
    snappy-c.cc
    snappy-stubs-internal.cc
    snappy-sinksource.cc
)

END()
