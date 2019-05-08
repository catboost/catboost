IF (PYTHON_CONFIG MATCHES "python3")
    EXPORTS_SCRIPT(catboost3.exports)
ELSE()
    EXPORTS_SCRIPT(catboost.exports)
ENDIF()

PYTHON_ADDINCL()

# add only headers for dynamic linking
ADDINCL(
    contrib/python/numpy/numpy/core/include
    contrib/python/numpy/numpy/core/src/common
    contrib/python/numpy/numpy/core/src/npymath
)

PEERDIR(
    catboost/libs/train_lib
    catboost/libs/algo
    catboost/libs/cat_feature
    catboost/libs/data_new
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/init
    catboost/libs/fstr
    catboost/libs/documents_importance
    catboost/libs/eval_result
    catboost/libs/helpers
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    catboost/libs/target
    library/containers/2d_array
    library/json/writer
)

SRCS(helpers.cpp)

# have to disable them because cython's numpy integration uses deprecated numpy API
NO_COMPILER_WARNINGS()

BUILDWITH_CYTHON_CPP(
    _catboost.pyx
    --module-name _catboost
)

IF (NOT OS_WINDOWS)
    ALLOCATOR(LF)
ELSE()
    ALLOCATOR(J)
ENDIF()

IF (OS_DARWIN)
    LDFLAGS(-headerpad_max_install_names)
ENDIF()
