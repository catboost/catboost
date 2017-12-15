LIBRARY()

NO_WERROR()



SRCS(
    cuda_base.cpp
    cuda_events_provider.cpp
    cuda_kernel_buffer.cpp
    gpu_memory_pool.cpp
    gpu_single_worker.cpp
    single_device.cpp
    device_provider.cpp
    cuda_manager.cpp
    cuda_buffer.cpp
    device_provider.cpp
    single_host_memory_copy_tasks.cpp
    kernel/arch.cu
    kernel/kernel.cu
    kernel/reduce.cu
    bandwidth_latency_calcer.cpp
    buffer_resharding.cpp
    mapping.cpp
    reduce.cpp
    cache.cpp
)

PEERDIR(
    library/binsaver
    library/threading/local_executor
    library/threading/future
    catboost/libs/logging
    catboost/libs/helpers
)


CUDA_NVCC_FLAGS(
    -std=c++11
    -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
    --ptxas-options=-v
)

END()
