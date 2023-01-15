UNITTEST()



NO_COMPILER_WARNINGS()

SRCS(
    messagext_ut.cpp
    test.proto
    ns.proto
)

PEERDIR(
    contrib/libs/protobuf
)

END()
