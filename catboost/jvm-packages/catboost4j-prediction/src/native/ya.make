

DLL(catboost4j-prediction exports exports.exports)

SRCS(
    ai_catboost_CatBoostJNIImpl.cpp
)

STRIP()

PEERDIR(
    catboost/libs/helpers
    catboost/libs/model
    contrib/libs/jdk
)

END()
