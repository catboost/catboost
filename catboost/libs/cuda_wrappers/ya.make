LIBRARY()




SRCS(
   base.cpp
   run_gpu_program.cpp
   cuda_graph.cpp
   cuda_event.cpp
   kernel.cu
   arch.cu
   stream_capture.cpp
)

PEERDIR(
    contrib/libs/cub
    catboost/libs/helpers
)

INCLUDE(${ARCADIA_ROOT}/catboost/libs/cuda_wrappers/default_nvcc_flags.make.inc)

END()
