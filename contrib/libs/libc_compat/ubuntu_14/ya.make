LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(BSD-3-Clause)



NO_PLATFORM()

NO_RUNTIME()

NO_UTIL()

DISABLE(NEED_PLATFORM_PEERDIRS)

IF (OS_SDK == "ubuntu-14")
    PEERDIR(
        build/platform/linux_sdk
    )
    SRCS(
        aligned_alloc.c
        c16rtomb.c
        c32rtomb.c
        getauxval.cpp
        mbrtoc16.c
        mbrtoc32.c
        secure_getenv.cpp
        timespec_get.c
    )
    SRC_CPP_PIC(
        glibc.cpp
        -fno-lto
    )
ENDIF()

END()
