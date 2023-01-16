

LIBRARY()

SRCS(
    case_insensitive_char_traits.cpp
    case_insensitive_string.cpp
)

PEERDIR(
    contrib/libs/libc_compat
    library/cpp/digest/murmur
)

END()

RECURSE_FOR_TESTS(ut)
