LIBRARY()

LICENSE(NCSA)

VERSION(2021-05-07)

ADDINCL(
    # Files are distributed between libcxxabi and libcxx in a weird manner
    # but we can not peerdir the latter to avoid loops (see below)
    # FIXME: sort includes open moving glibcxx-shims into its own dir
    contrib/libs/cxxsupp/libcxxabi-parts/include
    contrib/libs/cxxsupp/libcxx/include
)

# Do not create loop from libcxx, libcxxrt and libcxxabi-parts
NO_RUNTIME()

SRC_CPP_PIC(
    src/cxa_thread_atexit.cpp -fno-lto
)


SRCS(
    src/abort_message.cpp
)

END()
