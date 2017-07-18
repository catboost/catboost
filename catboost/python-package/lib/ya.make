PY_LIBRARY()



SRCDIR(catboost/python-package/catboost)

PEERDIR(
    catboost/libs/algo
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
