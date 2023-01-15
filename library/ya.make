

RECURSE(
    cpp
    langmask
    langmask/proto
    langmask/python
    langmask/serialization
    langmask/ut
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
    python
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
