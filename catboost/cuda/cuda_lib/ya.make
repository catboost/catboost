LIBRARY()

NO_WERROR()



SRCS(
    cuda_base.cpp
    cache.cpp
    cuda_events_provider.cpp
    cuda_kernel_buffer.cpp
    cuda_manager.cpp
    cuda_profiler.cpp
    cuda_buffer.cpp
    device_id.cpp
    device_provider.cpp
    devices_list.cpp
    gpu_single_worker.cpp
    inter_device_stream_section.cpp
    mapping.cpp
    memory_copy_performance.cpp
    single_device.cpp
    stream_section_tasks_launcher.cpp
    GLOBAL task.cpp
    worker_state.cpp
    hwloc_wrapper.cpp
    cuda_buffer_helpers/buffer_resharding.cpp
    GLOBAL cuda_buffer_helpers/reduce_scatter.cpp
    future/local_promise_future.cpp
    future/mpi_promise_future.cpp
    future/promise_factory.cpp
    kernel/arch.cu
    kernel/kernel.cu
    kernel/reduce.cu
    memory_pool/stack_like_memory_pool.cpp
    mpi/mpi_manager.cpp
    serialization/task_factory.cpp
    tasks_impl/cpu_func.cpp
    GLOBAL tasks_impl/enable_peers.cpp
    GLOBAL tasks_impl/host_tasks.cpp
    GLOBAL tasks_impl/kernel_task.cpp
    GLOBAL tasks_impl/memory_allocation.cpp
    GLOBAL tasks_impl/memory_state_func.cpp
    GLOBAL tasks_impl/request_stream_task.cpp
    GLOBAL tasks_impl/memory_copy_tasks.cpp
    tasks_impl/stream_section_task.cpp
    tasks_queue/mpi_task_queue.cpp
    tasks_queue/single_host_task_queue.cpp
)

PEERDIR(
    library/threading/local_executor
    library/threading/future
    catboost/libs/logging
    catboost/libs/helpers
    library/blockcodecs
)


CUDA_NVCC_FLAGS(
    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_61,code=compute_61
)


IF(USE_CUDA_MALLOC)
   CFLAGS(GLOBAL -DUSE_CUDA_MALLOC)
ENDIF()

IF(USE_MPI)
    CFLAGS(GLOBAL -DUSE_MPI)
    EXTRALIBS(-lmpi)

    IF(WITHOUT_CUDA_AWARE_MPI)
        CFLAGS(GLOBAL -DWITHOUT_CUDA_AWARE_MPI)
    ENDIF()

    IF(WRITE_MPI_MESSAGE_LOG)
         CFLAGS(GLOBAL -DWRITE_MPI_MESSAGE_LOG)
    ENDIF()
ENDIF()

IF(WITH_HWLOC)
    CFLAGS(GLOBAL -DWITH_HWLOC)
    EXTRALIBS(-lhwloc)
ENDIF()

END()
