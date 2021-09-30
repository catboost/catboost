LIBRARY()

WITHOUT_LICENSE_TEXTS()



LICENSE(
    Apache-2.0
    WITH
    LLVM-exception
)

VERSION(2021-06-07-87039c048c0cbc3d8cbba86187269b006bf2f373)

ADDINCL(
    contrib/libs/cxxsupp/libcxxabi/include
    contrib/libs/cxxsupp/libcxx/include
)

NO_COMPILER_WARNINGS()

NO_RUNTIME()

NO_UTIL()

CFLAGS(-D_LIBCXXABI_BUILDING_LIBRARY)

SRCDIR(contrib/libs/cxxsupp/libcxxabi)

SRCS(
    src/abort_message.cpp
    src/cxa_demangle.cpp
)

SRC_CPP_PIC(
    src/cxa_thread_atexit.cpp
    -fno-lto
)

END()
