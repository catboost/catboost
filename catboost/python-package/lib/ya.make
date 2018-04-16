PY_LIBRARY()



SRCDIR(catboost/python-package/catboost)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/documents_importance
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    library/containers/2d_array
    library/json/writer
    contrib/python/pandas
)

SRCS(catboost/python-package/catboost/helpers.cpp)

PY_SRCS(
    NAMESPACE catboost
    __init__.py
    version.py
    core.py
    datasets.py
    utils.py
    _catboost.pyx
    widget/__init__.py
    widget/ipythonwidget.py
    eval/_fold_model.py
    eval/_fold_models_handler.py
    eval/_fold_storage.py
    eval/_readers.py
    eval/_splitter.py
    eval/catboost_evaluation.py
    eval/evaluation_result.py
    eval/execution_case.py
    eval/factor_utils.py
    eval/log_config.py
    eval/utils.py
)

END()
