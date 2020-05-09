

LIBRARY()

SRCS(
    base64.cpp
)

PEERDIR(
    contrib/libs/base64/avx2
    contrib/libs/base64/ssse3
    contrib/libs/base64/neon32
    contrib/libs/base64/neon64
    contrib/libs/base64/plain32
    contrib/libs/base64/plain64
)

END()

RECURSE_FOR_TESTS(ut)
