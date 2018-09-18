LIBRARY()



SRCS(
    array_subset.cpp
    binarize_target.cpp
    clear_array.cpp
    compression.cpp
    cpu_random.cpp
    data_split.cpp
    dense_hash.cpp
    dense_hash_view.cpp
    element_range.cpp
    exception.cpp
    hash.cpp
    index_range.cpp
    interrupt.cpp
    map_merge.cpp
    matrix.cpp
    maybe_owning_array_holder.cpp
    mem_usage.cpp
    power_hash.cpp
    progress_helper.cpp
    permutation.cpp
    query_info_helper.cpp
    resource_constrained_executor.cpp
    resource_holder.cpp
    restorable_rng.cpp
    set.cpp
    vector_helpers.cpp
    wx_test.cpp
)

PEERDIR(
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/logging
    library/binsaver
    library/containers/2d_array
    library/digest/md5
    library/malloc/api
    library/threading/local_executor
    contrib/libs/clapack
)

END()
