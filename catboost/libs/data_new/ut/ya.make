UNITTEST_FOR(catboost/libs/data_new)



SRCS(
    columns_ut.cpp
    external_columns_ut.cpp
    features_layout_ut.cpp
    objects_grouping_ut.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/data_new
    catboost/libs/index_range
    catboost/libs/gpu_config/interface
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/quantization
)

END()
