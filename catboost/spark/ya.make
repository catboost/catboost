

IF (NOT OPENSOURCE OR USE_LOCAL_SWIG)
    RECURSE(
    catboost4j-spark/core/src/native_impl
)
ENDIF()

