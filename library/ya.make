

RECURSE(
    binsaver
    binsaver/ut_util
    binsaver/ut
    blockcodecs
    blockcodecs/fuzz
    blockcodecs/ut
    build_info
    charset
    charset/ut
    chromium_trace
    chromium_trace/benchmark
    chromium_trace/examples
    chromium_trace/ut
    comptrie
    comptrie/loader
    comptrie/loader/ut
    comptrie/ut
    comptrie/benchmark
    cpp
    dbg_output
    dbg_output/ut
    deprecated
    dns
    dns/ut
    enumbitset
    enumbitset/ut
    fast_exp
    fast_exp/benchmark
    fast_exp/ut
    fast_log
    getopt
    getopt/small
    http
    json
    json/flex_buffers
    json/flex_buffers/ut
    json/fuzzy_test
    json/ut
    json/writer/ut
    json/yson
    json/yson/ut
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
    logger
    logger/global
    logger/global/ut
    logger/ut
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
    resource/ut
    statistics
    statistics/ut
    svnversion
    svnversion/java
    testing
    text_processing
    threading
    token
    token/serialization
    token/serialization/ut
    token/ut
    unittest
    unittest/fat
    unittest/main
    unittest/ut
)

IF (HAVE_CUDA)
    RECURSE(
    cuda
)
ENDIF()

IF (OS_WINDOWS)
    RECURSE(
    
)
ELSE()
    RECURSE(
    
)
ENDIF()

IF (NOT SANITIZER_TYPE)
    RECURSE(
    
)
ENDIF()

NEED_CHECK()
