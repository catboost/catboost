PY_LIBRARY()



SRCDIR(catboost/python-package/catboost)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    library/containers/2d_array
    library/json/writer
)

SRCS(catboost/python-package/catboost/helpers.cpp)

PY_SRCS(
    NAMESPACE catboost
    __init__.py
    version.py
    core.py
    _catboost.pyx
    widget/__init__.py
    widget/ipythonwidget.py
)

END()
