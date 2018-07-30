PYMODULE(_catboost PREFIX "")



SRCDIR(
    catboost/python-package/catboost
)

SRCS(
    get_gpu_device_count_no_cuda.cpp
)

INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

END()
