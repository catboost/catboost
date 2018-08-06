LIBRARY(models)

NO_WERROR()



SRCS(
    kernel/add_model_value.cu
    GLOBAL add_bin_values.cpp
    add_oblivious_tree_model_doc_parallel.cpp
    oblivious_model.cpp
)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
