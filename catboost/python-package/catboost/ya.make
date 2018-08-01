PYMODULE(_catboost PREFIX "")



INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
    SRCS(
        get_gpu_device_count.cpp
    )
ELSE()
    SRCS(
        get_gpu_device_count_no_cuda.cpp
    )
ENDIF()

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        catboost//libs/for_python_package
    )
ENDIF()

END()
