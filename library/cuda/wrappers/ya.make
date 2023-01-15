LIBRARY()



SRCS(
    base.cpp
    exception.cpp
    run_gpu_program.cpp
    cuda_event.cpp
    kernel.cu
    kernel_helpers.cu
    arch.cu
    stream_pool.cpp
    cuda_vec.cpp
)

IF (CUDA_VERSION STREQUAL "10.1")
    SRCS(
        cuda_graph.cpp
        stream_capture.cpp
    )
ENDIF()

PEERDIR(
    contrib/libs/cub
    library/cpp/threading/future
)

INCLUDE(${ARCADIA_ROOT}/library/cuda/wrappers/default_nvcc_flags.make.inc)

END()
