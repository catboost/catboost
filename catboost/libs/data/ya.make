LIBRARY()



SRCS(
    load_data.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/logging
    library/containers/dense_hash
    library/digest/md5
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(column.h)

END()
