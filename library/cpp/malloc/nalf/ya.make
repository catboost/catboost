LIBRARY()



SRCS(
    alloc_helpers.h
    defs.h
    nalf_alloc.h
    nalf_alloc_cannibalizing_4k_cache.h
    nalf_alloc_chunkheader.h
    nalf_alloc_extmap.h
    nalf_alloc_impl.cpp
    nalf_alloc_impl.h
    nalf_alloc_pagepool.cpp
    nalf_alloc_pagepool.h
    malloc-info.cpp
)

PEERDIR(
    library/cpp/malloc/api
)

END()
