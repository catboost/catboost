

UNITTEST_FOR(library/json/yson)

ALLOCATOR(LF)

DATA(
    sbr://347179162
)

PEERDIR(
    library/histogram/simple
    web/app_host/lib/converter
)

SIZE(FAT)

TIMEOUT(600)

SRCS(
    json2yson_ut.cpp
)

END()
