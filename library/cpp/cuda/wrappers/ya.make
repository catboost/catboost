LIBRARY()



SRCS(
    base.cpp
    run_gpu_program.cpp
    cuda_event.cpp
    kernel.cu
    kernel_helpers.cu
    arch.cu
    stream_pool.cpp
    cuda_vec.cpp
)

IF (CUDA_VERSION VERSION_GE "10.1")
    SRCS(
        cuda_graph.cpp
        stream_capture.cpp
    )
ENDIF()

PEERDIR(
    contrib/libs/nvidia/cub
    library/cpp/cuda/exception
    library/cpp/threading/future
)

INCLUDE(${ARCADIA_ROOT}/library/cpp/cuda/wrappers/default_nvcc_flags.make.inc)

END()
