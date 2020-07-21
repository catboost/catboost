

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

INDUCED_DEPS(h
    ${ARCADIA_ROOT}/contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h
    ${ARCADIA_ROOT}/contrib/libs/flatbuffers/include/flatbuffers/flatbuffers_iter.h
)

END()
