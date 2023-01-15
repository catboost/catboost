LIBRARY()



ADDINCL(contrib/libs/flatbuffers/include)

PEERDIR(
    library/json
    contrib/libs/flatbuffers
)

SRCS(
    cvt.cpp
)

END()
