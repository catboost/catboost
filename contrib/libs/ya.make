

RECURSE(
    base64
    brotli
    clapack
    coreml
    crcutil
    cxxsupp/libcxx
    fastlz
    flatbuffers
    flatbuffers/samples
    fmath
    gamma_function_apache_math_port
    jemalloc
    libbz2
    libunwind_master
    libunwind_master/ut
    linuxvdso
    lz4
    lz4/generated
    lzmasdk
    nayuki_md5
    openssl
    openssl/apps
    openssl/dynamic
    platform
    platform/python
    protobuf
    protobuf/python
    protobuf/python/test
    protobuf/java
    protobuf/ut
    snappy
    sqlite3
    tensorboard
    yaml
    zlib
    zstd06
    zstd
)

IF (OS_FREEBSD OR OS_LINUX)
    RECURSE(
    
)
ENDIF()

IF (OS_DARWIN)
    RECURSE(
    
)
ENDIF()

IF (OS_LINUX)
    RECURSE(
    ibdrv
)
ENDIF()

IF (OS_WINDOWS)
    RECURSE(
    
)
ELSE()
    RECURSE(
    
)
ENDIF()

IF (OS_LINUX OR OS_WINDOWS)
    RECURSE(
    
)
ENDIF()

IF (OS_IOS)
    RECURSE(
    
)
ENDIF()

IF (OS_WINDOWS AND USE_UWP)
    # Other platforms will be added on demand or in background
    RECURSE(
    
)
ENDIF()

IF (OS_ANDROID)
    RECURSE(
    
)
ENDIF()

IF (ARCH_PPC64LE)
    RECURSE(
    
)
ENDIF()
