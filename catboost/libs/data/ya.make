LIBRARY()



SRCS(
    load_data.cpp
)

PEERDIR(
    library/digest/md5
    library/containers/dense_hash
    library/threading/local_executor
    catboost/libs/logging
)


GENERATE_ENUM_SERIALIZATION(column.h)


END()
