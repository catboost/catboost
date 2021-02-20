

DLL(catboost4j-prediction exports exports.exports)

SRCS(
    ai_catboost_CatBoostJNIImpl.cpp
)

STRIP()

PEERDIR(
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/model/model_export
)

IF (USE_SYSTEM_JDK)
    CFLAGS(-I${JAVA_HOME}/include)
    IF(OS_DARWIN)
        CFLAGS(-I${JAVA_HOME}/include/darwin)
    ELSEIF(OS_LINUX)
        CFLAGS(-I${JAVA_HOME}/include/linux)
    ELSEIF(OS_WINDOWS)
        CFLAGS(-I${JAVA_HOME}/include/win32)
    ENDIF()
ELSE()
    IF (NOT CATBOOST_OPENSOURCE OR AUTOCHECK)
        PEERDIR(contrib/libs/jdk)
    ELSE()
        # warning instead of an error to enable configure w/o specifying JAVA_HOME
        MESSAGE(WARNING System JDK required)
    ENDIF()
ENDIF()

END()
