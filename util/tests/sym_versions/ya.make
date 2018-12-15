PYTEST()



TEST_SRCS(test_glibc.py)

PEERDIR(
    library/python/resource
)

RESOURCE(
    ya.make /test_binaries
)

DEPENDS(
    # start binaries
    util/generic/ut
    util/charset/ut
    util/datetime/ut
    util/digest/ut
    util/draft/ut
    util/folder/ut
    util/memory/ut
    util/network/ut
    util/random/ut
    util/stream/ut
    util/string/ut
    util/system/ut
    util/thread/ut
    # end binaries
    contrib/python/pyelftools/readelf
)

FORK_SUBTESTS()

SPLIT_FACTOR(10)

END()
