

IF (NOT CATBOOST_OPENSOURCE)
    RECURSE(
    catboost4j-spark/core/src/native_impl
)
ENDIF()

