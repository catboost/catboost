LIBRARY()



LICENSE(Apache-2.0 WITH LLVM-exception)

ADDINCL(
    contrib/libs/cxxsupp/libcxxabi/include
    contrib/libs/cxxsupp/libcxx/include
)

NO_COMPILER_WARNINGS()

NO_RUNTIME()

NO_UTIL()

CFLAGS(
    -D_LIBCXXABI_BUILDING_LIBRARY
)

SRCDIR(
    contrib/libs/cxxsupp/libcxxabi
)

SRCS(
    src/abort_message.cpp
    src/cxa_demangle.cpp
)

SRC_CPP_PIC(
    src/cxa_thread_atexit.cpp -fno-lto
)

END()
