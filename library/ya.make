

RECURSE(
    accurate_accumulate
    accurate_accumulate/benchmark
    accurate_accumulate/benchmark/metrics
    binsaver
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
    colorizer
    colorizer/ut
    containers
    coroutine
    cppparser
    dbg_output
    dbg_output/ut
    diff
    diff/ut
    digest
    dns
    dns/ut
    dot_product
    dot_product/bench
    dot_product/ut
    fast_exp
    fast_exp/benchmark
    fast_exp/ut
    fast_log
    float16
    float16/ut
    getopt
    getopt/last_getopt_demo
    getopt/small
    getopt/ut
    grid_creator
    grid_creator/fuzz
    grid_creator/ut
    http
    json
    json/flex_buffers
    json/flex_buffers/ut
    json/fuzzy_test
    json/ut
    json/writer/ut
    json/yson
    json/yson/ut
    lcs
    lcs/ut
    lfalloc
    lfalloc/dbg
    lfalloc/dbg_info
    lfalloc/yt
    logger
    logger/global
    logger/global/ut
    logger/ut
    malloc
    neh
    neh/asio/ut
    neh/ut
    netliba
    object_factory
    object_factory/ut
    openssl
    par
    python
    resource
    resource/ut
    sse2neon
    statistics
    statistics/ut
    streams
    string_utils
    svnversion
    svnversion/java
    terminate_handler
    terminate_handler/sample
    threading
    unittest
    unittest/fat
    unittest/main
    unittest/ut
    yson
)

IF (OS_LINUX)
    RECURSE(
    
)
ENDIF()

IF (OS_WINDOWS)
    RECURSE(
    
)
ELSE()
    RECURSE(
    sse2neon/ut
)
ENDIF()

IF (NOT SANITIZER_TYPE)
    RECURSE(
    
)
ENDIF()

CHECK_DEPENDENT_DIRS(
    ALLOW_ONLY
    library
    contrib
    util
    yweb/config
)

NEED_CHECK()
