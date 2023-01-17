

LIBRARY()

LICENSE(Oracle)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

IF (OS_ANDROID)
    # Use NDK versions of jni headers
ELSEIF (OS_LINUX)
    ADDINCL(GLOBAL contrib/libs/jdk/include)
    ADDINCL(GLOBAL contrib/libs/jdk/include/linux)
ELSEIF (OS_DARWIN OR OS_IOS)
    ADDINCL(GLOBAL contrib/libs/jdk/include)
    ADDINCL(GLOBAL contrib/libs/jdk/include/darwin)
ELSEIF (OS_WINDOWS)
    ADDINCL(GLOBAL contrib/libs/jdk/include)
    ADDINCL(GLOBAL contrib/libs/jdk/include/win32)
ENDIF()

NO_UTIL()

END()
