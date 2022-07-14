FUZZ()



SRCS(
    fuzz.cpp
)

PEERDIR(
    library/cpp/containers/flat_hash/fuzz/fuzz_common
)

SIZE(LARGE)

TAG(ya:fat)

END()
