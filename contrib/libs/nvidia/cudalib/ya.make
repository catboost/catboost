LIBRARY()

# https://wiki.yandex-team.ru/devtools/cuda/



NO_RUNTIME()
NO_SANITIZE()

IF (NOT HAVE_CUDA)
    MESSAGE(FATAL_ERROR "No CUDA Toolkit for your build")
ENDIF()

IF (_USE_ARCADIA_CUDA)
    IF (HOST_OS_LINUX AND HOST_ARCH_X86_64)

        IF (OS_LINUX AND ARCH_X86_64)
            EXTERNAL_RESOURCE(CUDA sbr:235618970)
            CFLAGS(GLOBAL "-I$(CUDA)/include")
            LDFLAGS("-L$(CUDA)/lib64")

        ELSEIF (OS_LINUX AND ARCH_AARCH64)
            EXTERNAL_RESOURCE(CUDA sbr:152786617)
            CFLAGS(GLOBAL "-I$(CUDA)/targets/aarch64-linux/include")
            LDFLAGS("-L$(CUDA)/targets/aarch64-linux/lib")

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_DARWIN AND HOST_ARCH_X86_64)

        IF (OS_DARWIN AND ARCH_X86_64)
            EXTERNAL_RESOURCE(CUDA sbr:377468938)
            EXTERNAL_RESOURCE(CUDA_XCODE sbr:377517442)
            CFLAGS(GLOBAL "-I$(CUDA)/include")
            LDFLAGS("-L$(CUDA)/lib")

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSE()
        ENABLE(CUDA_NOT_FOUND)
    ENDIF()

    IF (CUDA_NOT_FOUND)
        MESSAGE(FATAL_ERROR "No CUDA Toolkit for the selected platform")
    ENDIF()

ELSE()
    IF (HOST_OS_WINDOWS)
        SET_APPEND_WITH_GLOBAL(USER_CFLAGS GLOBAL "\"-I${CUDA_ROOT}/include\"")
    ELSE()
        CFLAGS(GLOBAL "-I${CUDA_ROOT}/include")
    ENDIF()

    IF (HOST_OS_WINDOWS)
        SET_APPEND(LDFLAGS_GLOBAL "\"/LIBPATH:${CUDA_ROOT}/lib/x64\"")
    ELSEIF(HOST_OS_LINUX)
        LDFLAGS(-L${CUDA_ROOT}/lib64)
    ELSE()
        LDFLAGS(-L${CUDA_ROOT}/lib)
    ENDIF()
ENDIF()

IF (HOST_OS_WINDOWS)
    LDFLAGS(cublas.lib curand.lib cudart.lib cusparse.lib)
ELSE()
    IF (NOT PIC)
        EXTRALIBS(-lcublas_static -lcurand_static -lcudart_static -lcusparse_static -lculibos)
    ELSE()
        EXTRALIBS(-lcublas -lcurand -lcudart -lcusparse)
    ENDIF()
ENDIF()

END()
