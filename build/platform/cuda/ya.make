RESOURCES_LIBRARY()

# https://docs.yandex-team.ru/ya-make/manual/project_specific/cuda#cuda_host_compiler



IF (NOT HAVE_CUDA)
    MESSAGE(FATAL_ERROR "No CUDA Toolkit for your build")
ENDIF()

IF (USE_ARCADIA_CUDA)
    IF (HOST_OS_LINUX AND HOST_ARCH_X86_64)
        IF (OS_LINUX AND ARCH_X86_64)
            IF (CUDA_VERSION == "11.4")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:2410761119) # CUDA Toolkit 11.4.2 for Linux x86-64
            ELSEIF (CUDA_VERSION == "11.3")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:2213063565) # CUDA Toolkit 11.3.1 for Linux x86-64
            ELSEIF (CUDA_VERSION == "11.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:1882836946) # CUDA Toolkit 11.1.1 for Linux x86-64
            ELSEIF (CUDA_VERSION == "11.0")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:1647896014) # CUDA Toolkit 11.0.2 for Linux x86-64
            ELSEIF (CUDA_VERSION == "10.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:2077988857) # CUDA Toolkit 10.1.168 for Linux x86-64
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()
        ELSEIF(OS_LINUX AND ARCH_AARCH64)
            IF (CUDA_VERSION == "11.4")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:2410761119) # CUDA Toolkit 11.4.2 for linux-x64_64 (host part only)
                DECLARE_EXTERNAL_RESOURCE(CUDA_TARGET sbr:3840142733) # CUDA Toolkit 11.4.2 for linux-aarch64 (target part only)
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()
        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_LINUX AND HOST_ARCH_PPC64LE)
        IF (OS_LINUX AND ARCH_PPC64LE)
            IF (CUDA_VERSION == "10.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:1586537264) # CUDA Toolkit 10.1.168 for Linux ppc64le
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_DARWIN AND HOST_ARCH_X86_64)
        IF (OS_DARWIN AND ARCH_X86_64)
            IF (CUDA_VERSION == "10.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:978727023) # CUDA Toolkit 10.1.168 for macOS x86-64
            ENDIF()

        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_WINDOWS AND HOST_ARCH_X86_64)
        # CUDA on Windows does not support cross-compilation,
        # hence there is no need to divide it into HOST and TARGET resources.
        IF (OS_WINDOWS AND ARCH_X86_64)
            IF (CUDA_VERSION == "11.4")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:3866867639) # CDUA Toolkit 11.4.2 for windows-x86_64
            ELSEIF (CUDA_VERSION == "11.3")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:2215101513) # CUDA Toolkit 11.3.1 for Windows x86-64
            ELSEIF (CUDA_VERSION == "11.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:1896564605) # CUDA Toolkit 11.1.1 for Windows x86-64
            ELSEIF (CUDA_VERSION == "10.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA sbr:978734165) # CUDA Toolkit 10.1.168 for Windows x86-64
            ELSE()
                ENABLE(CUDA_NOT_FOUND)
            ENDIF()
        ELSE()
            ENABLE(CUDA_NOT_FOUND)
        ENDIF()

    ELSE()
        ENABLE(CUDA_NOT_FOUND)
    ENDIF()
ENDIF()

IF (USE_ARCADIA_CUDA_HOST_COMPILER)
    IF (HOST_OS_LINUX AND HOST_ARCH_X86_64)
        IF (OS_LINUX AND ARCH_X86_64)
            DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:1886578148) # Clang 11.0.0 for linux-x86_64
            IF (CUDA_VERSION VERSION_LT "11.2")
                # Equivalent to nvcc -allow-unsupported-compiler (present since 11.0).
                CFLAGS(GLOBAL "-D__NV_NO_HOST_COMPILER_CHECK")
            ENDIF()
        ELSEIF(OS_LINUX AND ARCH_AARCH64)
            DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:1886578148) # Clang 11.0.0 for linux-x86_64
        ELSE()
            ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_LINUX AND HOST_ARCH_PPC64LE)
        IF (OS_LINUX AND ARCH_PPC64LE)
            IF (CUDA_VERSION == "10.1")
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:1566513994) # Clang 7.0 for Linux ppc64le (not latest)
            ELSE()
                ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
            ENDIF()

        ELSE()
            ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_DARWIN AND HOST_ARCH_X86_64)
        IF (OS_DARWIN AND ARCH_X86_64)
            SET(__XCODE_RESOURCE_NAME CUDA_HOST_TOOLCHAIN)
            IF (CUDA_VERSION == "10.1")
                SET(__XCODE_TOOLCHAIN_VERSION "9.2") # (not latest)
            ELSE()
                SET(__XCODE_TOOLCHAIN_VERSION "")
                ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
            ENDIF()
            IF (__XCODE_TOOLCHAIN_VERSION)
                INCLUDE(${ARCADIA_ROOT}/build/platform/xcode/ya.make.inc)
            ENDIF()
        ELSE()
            ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
        ENDIF()

    ELSEIF (HOST_OS_WINDOWS AND HOST_ARCH_X86_64)
        IF (OS_WINDOWS AND ARCH_X86_64)
            # To create this toolchain, install MSVS on Windows and run:
            # devtools/tools_build/pack_sdk.py msvc out.tar
            # Note: it will contain patched "VC/Auxiliary/Build/vcvarsall.bat"
            # to prevent "nvcc fatal   : Host compiler targets unsupported OS."
            IF (CUDA_HOST_MSVC_VERSION == "14.28.29910")
                DECLARE_EXTERNAL_RESOURCE(CUDA_HOST_TOOLCHAIN sbr:2153212401)
            ELSE()
                MESSAGE(FATAL_ERROR "Unexpected or unspecified Microsoft Visual C++ CUDA host compiler version")
            ENDIF()

        ELSE()
            ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
        ENDIF()

    ELSE()
        ENABLE(CUDA_HOST_COMPILER_NOT_FOUND)
    ENDIF()
ENDIF()

IF (CUDA_NOT_FOUND)
    MESSAGE(FATAL_ERROR "No CUDA Toolkit for the selected platform")
ENDIF()

IF (CUDA_HOST_COMPILER_NOT_FOUND)
    MESSAGE(FATAL_ERROR "No CUDA host compiler for the selected platform and CUDA Toolkit version ${CUDA_VERSION}")
ENDIF()

# Use thrust and cub from Arcadia, not from HPC SDK
# NB:
#   it would be better to use PEERDIR instead,
#   but ymake does not allow PEERDIRs from RESOURCES_LIBRARY.
ADDINCL(
    GLOBAL contrib/libs/nvidia/thrust
    GLOBAL contrib/libs/nvidia/cub
)

IF (OS_WINDOWS)
    # Not using CFLAGS / LDFLAGS on Windows, as these macros do not allow spaces in variables
    # (and paths containing spaces are quite common on Windows)
    SET_APPEND_WITH_GLOBAL(USER_CFLAGS GLOBAL "\"-I${CUDA_ROOT}/include\"")
    SET_APPEND(LDFLAGS_GLOBAL "\"/LIBPATH:${CUDA_ROOT}/lib/x64\"")
ELSEIF (OS_LINUX AND ARCH_AARCH64)
    CFLAGS(GLOBAL "-I${CUDA_TARGET_RESOURCE_GLOBAL}/include")
    LDFLAGS(
        "-L${CUDA_TARGET_RESOURCE_GLOBAL}/lib"
        "-L${CUDA_TARGET_RESOURCE_GLOBAL}/lib/stubs"
    )
ELSEIF (OS_LINUX AND ARCH_X86_64)
    CFLAGS(GLOBAL "-I${CUDA_ROOT}/include")
    LDFLAGS(
        "-L${CUDA_ROOT}/lib64"
        "-L${CUDA_ROOT}/lib64/stubs"
    )
ELSEIF (OS_DARWIN)
    LDFLAGS("-L${CUDA_ROOT}/lib")
ELSE()
    MESSAGE(FATAL_ERROR "Unsupported target platform")
ENDIF()

IF (CUDA_REQUIRED)
    IF(HOST_OS_LINUX)
        EXTRALIBS(-lcuda)
    ELSEIF(HOST_OS_DARWIN)
        LDFLAGS("-F${CUDA_ROOT}/lib/stubs -framework CUDA")
    ENDIF()
ENDIF()

IF (HOST_OS_WINDOWS)
    LDFLAGS(cudadevrt.lib cudart_static.lib)
ELSE()
    EXTRALIBS(-lcudadevrt -lculibos)
    IF (USE_DYNAMIC_CUDA)
        EXTRALIBS(-lcudart)
    ELSE()
        EXTRALIBS(-lcudart_static)
    ENDIF()
ENDIF()

END()
