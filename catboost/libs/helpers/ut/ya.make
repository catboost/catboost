UNITTEST_FOR(catboost/libs/helpers)



SRCS(
    array_subset_ut.cpp
    checksum_ut.cpp
    compare_ut.cpp
    dbg_output_ut.cpp
    map_merge_ut.cpp
    math_utils_ut.cpp
    maybe_owning_array_holder_ut.cpp
    resource_constrained_executor_ut.cpp
    resource_holder_ut.cpp
    serialization_ut.cpp
)

PEERDIR(
    catboost/libs/helpers
)

END()
