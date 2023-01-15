

RECURSE(
    blockcodecs
    blockcodecs/fuzz
    blockcodecs/ut
    charset
    charset/ut
    chromium_trace
    chromium_trace/benchmark
    chromium_trace/examples
    chromium_trace/ut
    cpp
    enumbitset
    enumbitset/ut
    fast_exp
    fast_exp/benchmark
    fast_exp/ut
    langmask
    langmask/proto
    langmask/python
    langmask/serialization
    langmask/ut
    langs
    langs/ut
    lcs
    lcs/ut
    linear_regression
    linear_regression/benchmark
    linear_regression/ut
    neh
    neh/asio/ut
    neh/ut
    netliba
    object_factory
    object_factory/ut
    overloaded
    overloaded/ut
    packers
    packers/ut
    par
    python
    resource
    statistics
    statistics/ut
    svnversion
    svnversion/java
    testing
)

IF (HAVE_CUDA)
    RECURSE(
    cuda
)
ENDIF()

IF (NOT SANITIZER_TYPE)
    RECURSE(
    
)
ENDIF()

NEED_CHECK()
