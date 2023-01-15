LIBRARY()

NO_UTIL()



PEERDIR(
    library/cpp/malloc/api
    contrib/libs/tcmalloc
)
SRCS(
    malloc-info.cpp
)

END()
