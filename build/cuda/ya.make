RESOURCES_LIBRARY()

# https://wiki.yandex-team.ru/devtools/cuda/



IF (NOT HAVE_CUDA)
    MESSAGE(FATAL_ERROR "No CUDA Toolkit for your build")
ENDIF()

IF (USE_ARCADIA_CUDA)
    IF (HOST_OS_LINUX AND HOST_ARCH_X86_64)
        IF (OS_LINUX AND ARCH_X86_64)
            IF (CUDA_VERSION STREQUAL "9.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:494273996) # CUDA Toolkit 9.1.85.1 for Linux x86-64
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:243907179) # Clang 4.0 for Linux x86-64
            ELSEIF(CUDA_VERSION STREQUAL "9.0")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:562939434) # CUDA Toolkit 9.0.176.2 for Linux x86-64
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:133831678) # Clang 3.8 for Linux x86-64
            ELSEIF(CUDA_VERSION STREQUAL "8.0")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:494267287) # CUDA Toolkit 8.0.61.2 for Linux x86-64
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:133831678) # Clang 3.8 for Linux x86-64
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()

            CFLAGS(GLOBAL "-I$CUDA_RESOURCE_GLOBAL/include")
            LDFLAGS_FIXED("-L$CUDA_RESOURCE_GLOBAL/lib64")

        ELSEIF (OS_LINUX AND ARCH_AARCH64)
            DECLARE_EXTERNAL_RESOURCE(CUDA sbr:152786617) # 8.0.61
            CFLAGS(GLOBAL "-I$CUDA_RESOURCE_GLOBAL/targets/aarch64-linux/include")
            LDFLAGS_FIXED("-L$CUDA_RESOURCE_GLOBAL/targets/aarch64-linux/lib")

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_DARWIN AND HOST_ARCH_X86_64)
        IF (OS_DARWIN AND ARCH_X86_64)
            IF (CUDA_VERSION STREQUAL "9.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:494327636) # CUDA Toolkit 9.1.128 for macOS x86-64
                DECLARE_EXTERNAL_RESOURCE(CUDA_XCODE sbr:498971125) # Xcode 9.2
            ELSEIF (CUDA_VERSION STREQUAL "9.0")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:564419703) # CUDA Toolkit 9.0.176 for macOS x86-64
                DECLARE_EXTERNAL_RESOURCE(CUDA_XCODE sbr:500014407) # Xcode 8.2.1
            ELSE()
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:377468938)
                DECLARE_EXTERNAL_RESOURCE(CUDA_XCODE sbr:377517442)
            ENDIF()

            CFLAGS(GLOBAL "-I$CUDA_RESOURCE_GLOBAL/include")
            LDFLAGS_FIXED("-L$CUDA_RESOURCE_GLOBAL/lib")

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_WINDOWS AND HOST_ARCH_X86_64)
        IF (OS_WINDOWS AND ARCH_X86_64)
            IF (CUDA_VERSION STREQUAL "9.2")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:631278491) # CUDA Toolkit 9.2.148 for Windows 10 x86-64
            ELSEIF (CUDA_VERSION STREQUAL "9.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:636532525) # CUDA Toolkit 9.1.85.3 for Windows 10 x86-64
            ELSEIF (CUDA_VERSION STREQUAL "9.0")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:636578532) # CUDA Toolkit 9.0.176.4 for Windows 10 x86-64
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()

            IF (CUDA_HOST_MSVC_VERSION STREQUAL "14.11.25503")
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:637113754) # Microsoft Visual C++ 14.11.25503
            ELSEIF (CUDA_HOST_MSVC_VERSION STREQUAL "14.13.26128")
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:631304468) # Microsoft Visual C++ 14.13.26128
            ELSE()
                MESSAGE(FATAL_ERROR "Unexpected or unspecified Microsoft Visual C++ CUDA host compiler version")
            ENDIF()

            CFLAGS(GLOBAL "/I$CUDA_RESOURCE_GLOBAL/include")
            LDFLAGS_FIXED("/LIBPATH:$CUDA_RESOURCE_GLOBAL/lib/x64")

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
    EXTRALIBS(-lcublas_static -lcurand_static -lcudart_static -lcusparse_static -lculibos)
ENDIF()

END()
