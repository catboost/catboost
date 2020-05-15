

LIBRARY()

SRCS(
    cache.cpp
    cuda_base.cpp
    cuda_buffer.cpp
    cuda_buffer_helpers/buffer_resharding.cpp
    cuda_events_provider.cpp
    cuda_kernel_buffer.cpp
    cuda_manager.cpp
    cuda_profiler.cpp
    device_id.cpp
    device_provider.cpp
    devices_list.cpp
    future/local_promise_future.cpp
    future/mpi_promise_future.cpp
    future/promise_factory.cpp
    fwd.cpp
    GLOBAL cuda_buffer_helpers/reduce_scatter.cpp
    GLOBAL task.cpp
    GLOBAL tasks_impl/enable_peers.cpp
    GLOBAL tasks_impl/host_tasks.cpp
    GLOBAL tasks_impl/kernel_task.cpp
    GLOBAL tasks_impl/memory_allocation.cpp
    GLOBAL tasks_impl/memory_copy_tasks.cpp
    GLOBAL tasks_impl/memory_state_func.cpp
    GLOBAL tasks_impl/request_stream_task.cpp
    gpu_single_worker.cpp
    hwloc_wrapper.cpp
    inter_device_stream_section.cpp
    kernel/kernel.cu
    kernel/reduce.cu
    mapping.cpp
    memory_copy_performance.cpp
    memory_pool/stack_like_memory_pool.cpp
    mpi/mpi_manager.cpp
    serialization/task_factory.cpp
    single_device.cpp
    stream_section_tasks_launcher.cpp
    tasks_impl/cpu_func.cpp
    tasks_impl/stream_section_task.cpp
    tasks_queue/mpi_task_queue.cpp
    tasks_queue/single_host_task_queue.cpp
    worker_state.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
    library/blockcodecs
    library/cuda/wrappers
    library/cpp/threading/future
    library/cpp/threading/local_executor
    library/cpp/threading/name_guard
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

IF (USE_CUDA_MALLOC)
    CFLAGS(GLOBAL -DUSE_CUDA_MALLOC)
ENDIF()

IF (USE_MPI)
    CFLAGS(GLOBAL -DUSE_MPI)
    EXTRALIBS(-lmpi)
    IF (WITHOUT_CUDA_AWARE_MPI)
        CFLAGS(GLOBAL -DWITHOUT_CUDA_AWARE_MPI)
    ENDIF()
    IF (WRITE_MPI_MESSAGE_LOG)
        CFLAGS(GLOBAL -DWRITE_MPI_MESSAGE_LOG)
    ENDIF()
ENDIF()

IF (WITH_HWLOC)
    CFLAGS(GLOBAL -DWITH_HWLOC)
    EXTRALIBS(-lhwloc)
ENDIF()

END()
