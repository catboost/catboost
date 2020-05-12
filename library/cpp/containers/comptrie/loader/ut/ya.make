UNITTEST_FOR(library/cpp/containers/comptrie/loader)



ARCHIVE(
    NAME data.inc
    dummy.trie
)

SRCS(
    loader_ut.cpp
)

PEERDIR(
    library/comptrie/loader
)

END()
