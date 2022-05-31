LIBRARY()



SRCS(
    cgiparam.cpp
    cgiparam.h
)

PEERDIR(
    library/cpp/iterator
    library/cpp/string_utils/quote
    library/cpp/string_utils/scan
)

END()

RECURSE_FOR_TESTS(
    fuzz
    ut
)
