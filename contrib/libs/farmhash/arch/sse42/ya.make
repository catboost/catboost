LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(MIT)



NO_COMPILER_WARNINGS()

IF (NOT MSVC OR CLANG_CL)
    CFLAGS(-msse4.2)
ENDIF()

SRCDIR(contrib/libs/farmhash)

SRCS(
    farmhashsa.cc
    farmhashte.cc
)

END()
