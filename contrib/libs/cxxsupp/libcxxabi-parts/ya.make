LIBRARY()

LICENSE(NCSA)

VERSION(2021-05-07)

ADDINCL(
    contrib/libs/cxxsupp/libcxxabi-parts/include
)

SRC_CPP_PIC(
    src/cxa_thread_atexit.cpp -fno-lto
)

SRCS(
    src/abort_message.cpp
)

END()
