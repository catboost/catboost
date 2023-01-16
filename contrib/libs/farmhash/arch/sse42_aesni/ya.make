LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(MIT)



NO_COMPILER_WARNINGS()

IF (NOT MSVC OR CLANG_CL)
    CFLAGS(
        -msse4.2
        -maes
    )
ENDIF()

SRCDIR(contrib/libs/farmhash)

SRCS(
    farmhashsu.cc
)

END()
