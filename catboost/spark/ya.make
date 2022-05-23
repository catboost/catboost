

IF (NOT OPENSOURCE OR USE_LOCAL_SWIG OR EXPORT_CMAKE)
    RECURSE(
    catboost4j-spark/core/src/native_impl
)
ENDIF()

