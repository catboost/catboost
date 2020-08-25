

IF (NOT CATBOOST_OPENSOURCE)
    RECURSE(
    catboost4j-spark/src/native_impl
)
ENDIF()

