

UNITTEST_FOR(library/json/yson)

ALLOCATOR(LF)

DATA(
    sbr://363537653
)

PEERDIR(
    library/blockcodecs
    library/histogram/simple
    library/unittest
)

SIZE(FAT)

TIMEOUT(600)

SRCS(
    json2yson_ut.cpp
)

END()
