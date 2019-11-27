UNITTEST_FOR(catboost/libs/helpers)



SIZE(MEDIUM)

SRCS(
    array_subset_ut.cpp
    checksum_ut.cpp
    compression_ut.cpp
    dbg_output_ut.cpp
    double_array_iterator_ut.cpp
    dynamic_iterator_ut.cpp
    guid_ut.cpp
    map_merge_ut.cpp
    math_utils_ut.cpp
    maybe_owning_array_holder_ut.cpp
    permutation_ut.cpp
    polymorphic_type_containers_ut.cpp
    quantile_ut.cpp
    resource_constrained_executor_ut.cpp
    resource_holder_ut.cpp
    sample_ut.cpp
    serialization_ut.cpp
    short_vector_ops_ut.cpp
    sparse_array_ut.cpp
    wx_test_ut.cpp
    xml_output_ut.cpp
)

PEERDIR(
    catboost/libs/helpers
    library/binsaver/ut_util
)

END()
