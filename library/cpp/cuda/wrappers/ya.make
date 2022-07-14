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
    cuda_graph.cpp
    stream_capture.cpp
)

PEERDIR(
    contrib/libs/nvidia/cub
    library/cpp/cuda/exception
    library/cpp/threading/future
)

INCLUDE(${ARCADIA_ROOT}/library/cpp/cuda/wrappers/default_nvcc_flags.make.inc)

END()
