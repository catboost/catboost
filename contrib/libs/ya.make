

RECURSE(
    base64
    brotli
    clapack
    coreml
    cppdemangle
    crcutil
    cxxsupp/libcxx
    cxxsupp/libcxx-filesystem
    cxxsupp/libcxxabi-parts
    expat
    fastlz
    flatbuffers
    fmath
    gamma_function_apache_math_port
    jdk
    jemalloc
    jemalloc/dynamic
    libbz2
    libc_compat
    libunwind
    linux-headers
    linuxvdso
    lz4
    lz4/generated
    lzmasdk
    nayuki_md5
    onnx
    openssl
    protobuf
    protobuf/python
    pugixml
    python
    python/ut
    r-lang
    re2
    snappy
    sqlite3
    tbb
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

IF (OS_ANDROID)
    RECURSE(
    
)
ENDIF()

IF (OS_IOS AND ARCH_ARM64 OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()

IF (OS_LINUX AND ARCH_ARM64)
    RECURSE(
    
)
ENDIF()
