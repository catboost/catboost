PY_LIBRARY()



PY_SRCS(
    __init__.py
    _fold_model.py
    _fold_models_handler.py
    _fold_storage.py
    _readers.py
    _splitter.py
    catboost_evaluation.py
    evaluation_result.py
    execution_case.py
    factor_utils.py
    log_config.py
    utils.py
)

PEERDIR(
    contrib/python/pandas
    contrib/python/scipy-0.18.1/scipy/stats
    catboost/python-package/lib
)

END()
