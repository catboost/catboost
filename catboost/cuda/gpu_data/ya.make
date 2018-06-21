LIBRARY()

NO_WERROR()



SRCS(
    batch_binarized_ctr_calcer.cpp
    cat_features_dataset.cpp
    compressed_index.cpp
    compressed_index_builder.cpp
    dataset_helpers.cpp
    feature_parallel_dataset.cpp
    doc_parallel_dataset_builder.cpp
    feature_parallel_dataset_builder.cpp
    gpu_grid_creator.cpp
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
)

PEERDIR(
    library/grid_creator
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/ctrs
    catboost/cuda/utils

)

CUDA_NVCC_FLAGS(
   --expt-relaxed-constexpr
   -gencode arch=compute_30,code=compute_30
   -gencode arch=compute_35,code=sm_35
   -gencode arch=compute_50,code=compute_50
   -gencode arch=compute_52,code=sm_52
   -gencode arch=compute_60,code=compute_60
   -gencode arch=compute_61,code=compute_61
   -gencode arch=compute_61,code=sm_61
   -gencode arch=compute_70,code=sm_70
   -gencode arch=compute_70,code=compute_70
)

GENERATE_ENUM_SERIALIZATION(grid_policy.h)

END()
