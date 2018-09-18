LIBRARY(targets)

NO_WERROR()



SRCS(
    kernel/pointwise_targets.cu
    kernel/query_rmse.cu
    kernel/query_softmax.cu
    kernel/pair_logit.cu
    kernel/yeti_rank_pointwise.cu
    kernel/pfound_f.cu
    kernel/query_cross_entropy.cu
    kernel/multilogit.cu
    target_func.cpp
    pfound_f.cpp
    querywise_targets_impl.cpp
    pointwise_target_impl.cpp
    multiclass_targets.cpp
    pair_logit_pairwise.cpp
    gpu_metrics.cpp
    auc.cpp
    query_cross_entropy.cpp
    GLOBAL kernel.cpp
    GLOBAL query_cross_entropy_kernels.cpp

)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
    catboost/libs/helpers
    catboost/libs/options
    catboost/libs/metrics
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
