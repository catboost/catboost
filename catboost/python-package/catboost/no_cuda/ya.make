PY_ANY_MODULE(_catboost PREFIX "")



SRCDIR(
    catboost/python-package/catboost
)

PEERDIR(
    catboost/libs/gpu_config/force_no_cuda
)

INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

END()
