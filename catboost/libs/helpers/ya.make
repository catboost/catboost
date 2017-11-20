LIBRARY()



SRCS(
    dense_hash.cpp
    dense_hash_view.cpp
    power_hash.cpp
    progress_helper.cpp
    matrix.cpp
    interrupt.cpp
    eval_helpers.cpp
)

PEERDIR(
    catboost/libs/logging
    library/binsaver
    library/containers/2d_array
    library/digest/md5
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
