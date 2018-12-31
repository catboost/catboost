LIBRARY()



PEERDIR(
    catboost/libs/logging
    catboost/libs/gpu_config/interface
)

IF(HAVE_CUDA)
    PEERDIR(
        build/platform/cuda
    )
    SRCS(
        get_gpu_device_count_cuda.cpp
    )
ELSE()
    SRCS(
        get_gpu_device_count_no_cuda.cpp
    )
ENDIF()

END()
