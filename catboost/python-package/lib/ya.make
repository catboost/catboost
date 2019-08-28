PY23_LIBRARY()



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
    catboost/libs/hyperparameter_tuning
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
    contrib/python/graphviz
    contrib/python/numpy
    contrib/python/pandas
)

IF(PYTHON2)
    PEERDIR(
        contrib/python/enum34
    )
ENDIF()

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        catboost//libs/for_python_package
        contrib/python/matplotlib
        contrib/python/plotly
    )
ENDIF()

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

# have to disable them because cython's numpy integration uses deprecated numpy API
NO_COMPILER_WARNINGS()

# In case of android with python3 there will be the following error: "fatal error: 'crypt.h' file not found"
IF(NOT OS_ANDROID OR PYTHON2)
    SRCS(catboost/python-package/catboost/helpers.cpp)

    NO_CHECK_IMPORTS(
        catboost.widget.*
    )

    PY_SRCS(
        NAMESPACE catboost
        __init__.py
        version.py
        core.py
        datasets.py
        utils.py
        dev_utils.py
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
ENDIF()

NO_LINT()

END()
