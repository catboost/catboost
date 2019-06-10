UNITTEST_FOR(catboost/libs/helpers)



SIZE(MEDIUM)

SRCS(
    array_subset_ut.cpp
    checksum_ut.cpp
    compare_ut.cpp
    dbg_output_ut.cpp
    map_merge_ut.cpp
    math_utils_ut.cpp
    maybe_owning_array_holder_ut.cpp
    permutation_ut.cpp
    resource_constrained_executor_ut.cpp
    resource_holder_ut.cpp
    serialization_ut.cpp
    vec_list_ut.cpp
    wx_test_ut.cpp
    xml_output_ut.cpp
)

PEERDIR(
    catboost/libs/helpers
)

END()
