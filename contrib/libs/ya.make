

RECURSE(
    base64
    brotli
    clapack
    coreml
    cppdemangle/all
    crcutil
    cxxsupp/libcxx
    fastlz
    flatbuffers
    flatbuffers/samples
    fmath
    gamma_function_apache_math_port
    jdk
    jemalloc
    libbz2
    libunwind_master
    libunwind_master/ut
    linuxvdso
    lz4
    lz4/generated
    lzmasdk
    musl-1.1.20
    nayuki_md5
    onnx
    openssl
    openssl/apps
    openssl/dynamic
    protobuf
    protobuf/java
    protobuf/python
    protobuf/python/test
    protobuf/ut
    python
    python/ut
    snappy
    sqlite3
    tensorboard
    zlib
    zstd
    zstd06
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
