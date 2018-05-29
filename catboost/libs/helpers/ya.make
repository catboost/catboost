LIBRARY()



SRCS(
    dense_hash.cpp
    dense_hash_view.cpp
    power_hash.cpp
    progress_helper.cpp
    matrix.cpp
    interrupt.cpp
    eval_helpers.cpp
    permutation.cpp
    restorable_rng.cpp
    binarize_target.cpp
    query_info_helper.cpp
    data_split.cpp
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

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
