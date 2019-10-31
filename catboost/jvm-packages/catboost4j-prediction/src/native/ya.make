

DLL(catboost4j-prediction exports exports.exports)

SRCS(
    ai_catboost_CatBoostJNIImpl.cpp
)

STRIP()

PEERDIR(
    catboost/libs/helpers
    catboost/libs/model
)

IF (USE_SYSTEM_JDK)
    CFLAGS(-I${USE_SYSTEM_JDK}/include)
    IF(OS_DARWIN)
        CFLAGS(-I${USE_SYSTEM_JDK}/include/darwin)
    ELSEIF(OS_LINUX)
        CFLAGS(-I${USE_SYSTEM_JDK}/include/linux)
    ELSEIF(OS_WINDOWS)
        CFLAGS(-I${USE_SYSTEM_JDK}/include/win32)
    ENDIF()
ELSE()
    IF (NOT CATBOOST_OPENSOURCE OR AUTOCHECK)
        PEERDIR(contrib/libs/jdk)
    ELSE()
        MESSAGE(WARNING System JDK required)
    ENDIF()
ENDIF()

END()
