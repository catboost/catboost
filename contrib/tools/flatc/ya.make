

PROGRAM()

NO_UTIL()

ADDINCL(
    contrib/libs/flatbuffers/include
)

PEERDIR(
    contrib/libs/flatbuffers/flatc
)

SRCDIR(
    contrib/libs/flatbuffers/src
)

SRCS(
    flatc_main.cpp
)

END()
