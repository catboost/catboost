

UNITTEST_FOR(library/string_utils/base64)

SRCS(
    base64_ut.cpp
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
