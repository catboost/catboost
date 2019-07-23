LIBRARY()



PEERDIR(
    library/python/symbols/registry
)

SRCS(
    GLOBAL syms.cpp
)

END()

RECURSE_FOR_TESTS(ut)
