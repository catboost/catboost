IF (PYTHON_CONFIG MATCHES "python3")
    PYMODULE(_catboost EXPORTS catboost3.exports PREFIX "")
ELSE()
    PYMODULE(_catboost EXPORTS catboost.exports PREFIX "")
ENDIF()



PYTHON_ADDINCL()

PEERDIR(
    catboost/libs/algo
)

SRCS(helpers.cpp)

BUILDWITH_CYTHON_CPP(
    _catboost.pyx
    --module-name _catboost
)

END()
