LIBRARY()



SRCS(
    batch_binarized_ctr_calcer.cpp
    cat_features_dataset.cpp
    compressed_index.cpp
    compressed_index_builder.cpp
    dataset_helpers.cpp
    feature_parallel_dataset.cpp
    doc_parallel_dataset_builder.cpp
    doc_parallel_dataset.cpp
    feature_parallel_dataset_builder.cpp
    kernel/split.cu
    kernel/query_helper.cu
    kernel/binarize.cu
    GLOBAL kernels.cpp
    GLOBAL splitter.cpp
    oblivious_tree_bin_builder.cpp
    pinned_memory_estimation.cpp
    samples_grouping.cpp
    samples_grouping_gpu.cpp
    querywise_helper.cpp
    bootstrap.cpp
    non_zero_filter.cpp
    ctr_helper.cpp
    feature_layout.cpp
    gpu_binarization_helpers.cpp
    estimated_features_calcer.cpp
)

PEERDIR(
    library/cpp/grid_creator
    library/cpp/threading/local_executor
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/ctrs
    catboost/private/libs/ctr_description
    catboost/libs/data
    catboost/libs/helpers
    catboost/private/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

GENERATE_ENUM_SERIALIZATION(grid_policy.h)

END()
