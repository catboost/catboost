LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_RUNTIME()

NO_COMPILER_WARNINGS()

IF (NOT USE_STL_SYSTEM)
    ADDINCL(
        contrib/libs/cxxsupp/libcxx/include
    )
ENDIF()

CFLAGS(
    -D_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS
)

SRCS(
    cxa_demangle.cpp
)

END()

RECURSE(
    filt
    fuzz
)

RECURSE_FOR_TESTS(
    ut
)
