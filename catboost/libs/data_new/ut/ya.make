UNITTEST_FOR(catboost/libs/data_new)



SRCS(
    borders_io_ut.cpp
    columns_ut.cpp
    data_provider_ut.cpp
    dsv_parser_ut.cpp
    external_columns_ut.cpp
    features_layout_ut.cpp
    load_data_from_dsv_ut.cpp
    meta_info_ut.cpp
    model_dataset_compatibility_ut.cpp
    objects_grouping_ut.cpp
    objects_ut.cpp
    order_ut.cpp
    process_data_blocks_from_dsv_ut.cpp
    quantization_ut.cpp
    target_ut.cpp
    unaligned_mem_ut.cpp
    util.cpp
    weights_ut.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/data_new
    catboost/libs/data_new/ut/lib
    catboost/libs/index_range
    catboost/libs/gpu_config/interface
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/quantization
)

END()
