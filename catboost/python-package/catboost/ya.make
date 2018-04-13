PYMODULE(_catboost PREFIX "")
IF (PYTHON_CONFIG MATCHES "python3")
    EXPORTS_SCRIPT(catboost3.exports)
ELSE()
    EXPORTS_SCRIPT(catboost.exports)
ENDIF()

USE_LINKER_GOLD()



PYTHON_ADDINCL()

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

PEERDIR(
    catboost/libs/train_lib
    catboost/libs/algo
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
)

SRCS(helpers.cpp)

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

END()
