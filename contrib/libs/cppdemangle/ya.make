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

IF (MUSL)
    ADDINCL(
        contrib/libs/musl/arch/x86_64
        contrib/libs/musl/arch/generic
        contrib/libs/musl/include
        contrib/libs/musl/extra
    )
ENDIF ()

END()

RECURSE(
    filt
    fuzz
)

RECURSE_FOR_TESTS(
    ut
)
