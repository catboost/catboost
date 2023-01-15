UNITTEST_FOR(util)



SRCS(
    ysaveload_ut.cpp
)

END()

RECURSE_ROOT_RELATIVE(
    util/charset/ut
    util/datetime/ut
    util/digest/ut
    util/draft/ut
    util/folder/ut
    util/generic/ut
    util/memory/ut
    util/network/ut
    util/random/ut
    util/stream/ut
    util/string/ut
    util/system/ut
    util/thread/ut
)
