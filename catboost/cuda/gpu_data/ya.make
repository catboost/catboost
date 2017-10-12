LIBRARY()

NO_WERROR()



SRCS(
    cat_features_dataset.cpp
    binarized_dataset.cpp
    binarized_dataset_builder.cpp
    fold_based_dataset.cpp
    fold_based_dataset_builder.cpp
    gpu_grid_creator.cpp
    kernel/split.cu
    kernel/binarize.cu
    remote_binarize.cpp
    splitter.cpp
    oblivious_tree_bin_builder.cpp
    pinned_memory_estimation.cpp
)

PEERDIR(
    library/grid_creator
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/ctrs
)

CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr
    -std=c++11
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
    --ptxas-options=-v
)

END()
