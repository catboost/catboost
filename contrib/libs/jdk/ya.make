

LIBRARY()

LICENSE(Oracle)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

ADDINCL(GLOBAL contrib/libs/jdk/include)

IF (OS_LINUX OR OS_ANDROID)
    ADDINCL(GLOBAL contrib/libs/jdk/include/linux)
ELSEIF (OS_DARWIN OR OS_IOS)
    ADDINCL(GLOBAL contrib/libs/jdk/include/darwin)
ELSEIF (OS_WINDOWS)
    ADDINCL(GLOBAL contrib/libs/jdk/include/win32)
ENDIF()

NO_UTIL()

END()
