LIBRARY()



SRCS(
    load_data.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/logging
    catboost/libs/column_description
    library/containers/dense_hash
    library/digest/md5
    library/threading/local_executor
)

END()
