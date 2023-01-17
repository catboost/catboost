LIBRARY()

#!!!


PEERDIR(
    contrib/libs/crcutil
)

SRCS(
    crc32c.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
