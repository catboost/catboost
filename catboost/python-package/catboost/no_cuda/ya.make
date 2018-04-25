PYMODULE(_catboost PREFIX "")



SRCDIR(
    catboost/python-package/catboost
)

INCLUDE(
    ${ARCADIA_ROOT}/catboost/python-package/catboost/ya.make.inc
)

END()
