LIBRARY(models)



SRCS(
    kernel/add_model_value.cu
    GLOBAL add_bin_values.cpp
    add_oblivious_tree_model_doc_parallel.cpp
    oblivious_model.cpp
    GLOBAL add_region_doc_parallel.cpp
    region_model.cpp
    non_summetric_tree.cpp
    GLOBAL add_non_symmetric_tree_doc_parallel.cpp
    model_converter.cpp
    compact_model.cpp
)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
