LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(BSD-3-Clause)

VERSION(2.0)

ORIGINAL_SOURCE(https://github.com/gperftools/gperftools)



NO_UTIL()

NO_RUNTIME()

IF (OS_LINUX)
    PEERDIR(
        contrib/libs/linuxvdso/original
    )
    SRCS(
        interface.cpp
    )
ELSE()
    SRCS(
        fake.cpp
    )
ENDIF()

END()
