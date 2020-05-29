

RECURSE(
    android_ifaddrs
    base64
    brotli
    clapack
    coreml
    cppdemangle
    crcutil
    cxxsupp/libcxx
    cxxsupp/libcxx-filesystem
    expat
    fastlz
    flatbuffers
    flatbuffers/samples
    fmath
    gamma_function_apache_math_port
    jemalloc
    libbz2
    libunwind
    linux-headers
    linuxvdso
    lz4
    lz4/generated
    lzmasdk
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
    pugixml
    python
    python/ut
    r-lang
    re2
    snappy
    sqlite3
    tensorboard
    xxhash
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

    IF (NOT OS_SDK STREQUAL "ubuntu-12")
        RECURSE(
    
)
    ENDIF()
ENDIF()

IF (OS_WINDOWS)
    RECURSE(
    
)
ELSE()
    RECURSE(
    re2/tests
)
ENDIF()

IF (OS_LINUX OR OS_WINDOWS)
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

IF (OS_IOS AND ARCH_ARM64 OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()
