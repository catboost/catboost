

RECURSE(
    base64
    brotli
    clapack
    coreml
    crcutil
    cxxsupp/libcxx
    cxxsupp/libcxxabi-parts
    double-conversion
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
    lzma
    lzmasdk
    mimalloc
    nayuki_md5
    nvidia
    onnx
    onnx/proto
    onnx/python
    openssl
    protobuf
    protobuf/python
    pugixml
    python
    python/ut
    qhull
    r-lang
    re2
    snappy
    sqlite3
    tbb
    tcmalloc
    tcmalloc/dynamic
    tensorboard
    xxhash
    zlib
    zstd
    zstd06
)

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

IF (MUSL)
    RECURSE(
    
)
ENDIF()
