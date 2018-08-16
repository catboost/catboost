UNITTEST_FOR(catboost/libs/helpers)



SRCS(
    array_subset_ut.cpp
    map_merge_ut.cpp
    maybe_owning_array_holder_ut.cpp
    resource_constrained_executor_ut.cpp
    resource_holder_ut.cpp
)

PEERDIR(
    catboost/libs/helpers
)

END()
