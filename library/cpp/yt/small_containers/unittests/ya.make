GTEST(unittester-small-containers)



SRCS(
    compact_flat_map_ut.cpp
    compact_set_ut.cpp
    compact_vector_ut.cpp
)

PEERDIR(
    library/cpp/yt/small_containers
    library/cpp/testing/gtest
)

END()
