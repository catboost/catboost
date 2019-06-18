LIBRARY()




SRCS(
   base.cpp
   run_gpu_program.cpp
   cuda_event.cpp
   kernel.cu
   arch.cu
)
IF (CUDA_VERSION STREQUAL "10.1")
    SRCS(
        cuda_graph.cpp
        stream_capture.cpp
    )
ENDIF()

PEERDIR(
    contrib/libs/cub
    catboost/libs/helpers
    library/threading/future
)

INCLUDE(${ARCADIA_ROOT}/catboost/libs/cuda_wrappers/default_nvcc_flags.make.inc)

END()
