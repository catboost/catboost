LIBRARY()



PEERDIR(
    catboost/libs/gpu_config/interface
)

SRCS(
    get_gpu_device_count_no_cuda.cpp
)

END()
