PYMODULE(_catboost PREFIX "")



INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

PEERDIR(
    catboost/libs/gpu_config/maybe_have_cuda
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        catboost//libs/for_python_package
    )
ENDIF()

NO_LINT()

END()
