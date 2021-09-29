LIBRARY()

NO_UTIL()



PEERDIR(
    library/cpp/malloc/api
    contrib/libs/tcmalloc/malloc_extension
)
SRCS(
    malloc-info.cpp
)

END()
