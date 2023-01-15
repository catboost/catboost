LIBRARY()



ADDINCL(contrib/libs/flatbuffers/include)

PEERDIR(
    library/cpp/json
    contrib/libs/flatbuffers
)

SRCS(
    cvt.cpp
)

END()
