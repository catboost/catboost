LIBRARY()



SRCS(
    delim_string_iter.cpp
    split_iterator.cpp
)

PEERDIR(
    library/cpp/deprecated/kmp
)

END()

RECURSE_FOR_TESTS(
    ut
)
