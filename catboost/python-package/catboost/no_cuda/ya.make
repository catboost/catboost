PY_ANY_MODULE(_catboost PREFIX "")
CMAKE_EXPORTED_TARGET_NAME(_catboost_no_cuda)



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
