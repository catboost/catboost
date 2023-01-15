

UNITTEST_FOR(library/cpp/json/yson)

ALLOCATOR(LF)

DATA(sbr://363537653)

PEERDIR(
    library/blockcodecs
    library/cpp/histogram/simple
    library/cpp/unittest
)

SIZE(LARGE)

TAG(ya:fat)

TIMEOUT(600)

SRCS(
    json2yson_ut.cpp
)

END()
