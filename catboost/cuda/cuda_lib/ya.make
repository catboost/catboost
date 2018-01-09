LIBRARY()

NO_WERROR()



SRCS(
    cuda_base.cpp
    cuda_events_provider.cpp
    cuda_kernel_buffer.cpp
    memory_pool/stack_like_memory_pool.cpp
    gpu_single_worker.cpp
    single_device.cpp
    device_provider.cpp
    cuda_manager.cpp
    cuda_buffer.cpp
    device_provider.cpp
    tasks_impl/single_host_memory_copy_tasks.cpp
    kernel/arch.cu
    kernel/kernel.cu
    kernel/reduce.cu
    bandwidth_latency_calcer.cpp
    cuda_buffer_helpers/buffer_resharding.cpp
    mapping.cpp
    cuda_buffer_helpers/reduce_scatter.cpp
    cache.cpp
    tasks_queue/single_host_task_queue.cpp
)

PEERDIR(
    library/threading/local_executor
    library/threading/future
    catboost/libs/logging
    catboost/libs/helpers
)


CUDA_NVCC_FLAGS(
    -std=c++11
    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
    --ptxas-options=-v
)


IF(CB_USE_CUDA_MALLOC)
   CFLAGS(GLOBAL -DCB_USE_CUDA_MALLOC)
ENDIF()

END()
