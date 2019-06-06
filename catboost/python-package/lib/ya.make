PY_LIBRARY()



SRCDIR(catboost/python-package/catboost)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/cat_feature
    catboost/libs/data_new
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/fstr
    catboost/libs/gpu_config/maybe_have_cuda
    catboost/libs/documents_importance
    catboost/libs/eval_result
    catboost/libs/helpers
    catboost/libs/init
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantized_pool_analysis
    catboost/libs/target
    library/containers/2d_array
    library/json/writer
    contrib/python/enum34
    contrib/python/graphviz
    contrib/python/numpy
    contrib/python/pandas
)

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        catboost//libs/for_python_package
        contrib/python/matplotlib
        contrib/python/plotly
    )
ENDIF()

SRCS(catboost/python-package/catboost/helpers.cpp)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

# have to disable them because cython's numpy integration uses deprecated numpy API
NO_COMPILER_WARNINGS()

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

NO_LINT()

END()
