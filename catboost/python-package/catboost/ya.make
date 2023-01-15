PY_ANY_MODULE(_catboost PREFIX "")



INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

PEERDIR(
    catboost/libs/gpu_config/maybe_have_cuda
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        catboost//private/libs/for_python_package
    )
ENDIF()

NO_LINT()

END()
