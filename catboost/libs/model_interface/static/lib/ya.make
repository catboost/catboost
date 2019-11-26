

LIBRARY()

SRCDIR(catboost/libs/model_interface)

SRCS(
    c_api.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/model
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/libs/model/cuda
    )
ENDIF()

CFLAGS(-DCATBOOST_API_STATIC_LIB)

END()

