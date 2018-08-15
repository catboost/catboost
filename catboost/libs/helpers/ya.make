LIBRARY()



SRCS(
    array_subset.h
    binarize_target.cpp
    data_split.cpp
    dense_hash.cpp
    dense_hash_view.cpp
    index_range.h
    interrupt.cpp
    map_merge.cpp
    matrix.cpp
    pairs_util.cpp
    power_hash.cpp
    progress_helper.cpp
    permutation.cpp
    query_info_helper.cpp
    resource_constrained_executor.cpp
    restorable_rng.cpp
    wx_test.cpp
)

PEERDIR(
    catboost/libs/data_util
    catboost/libs/logging
    catboost/libs/options
    library/binsaver
    library/containers/2d_array
    library/digest/md5
    library/malloc/api
    library/threading/local_executor
    contrib/libs/clapack
)

END()
