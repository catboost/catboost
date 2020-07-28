

UNITTEST_FOR(library/cpp/json/yson)

ALLOCATOR(LF)

DATA(sbr://363537653)

PEERDIR(
    library/cpp/blockcodecs
    library/cpp/histogram/simple
    library/cpp/testing/unittest
)

SIZE(LARGE)

TAG(ya:fat)

TIMEOUT(600)

SRCS(
    json2yson_ut.cpp
)

END()
