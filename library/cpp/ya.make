

RECURSE(
    accurate_accumulate
    accurate_accumulate/benchmark
    accurate_accumulate/benchmark/metrics
    archive/ut
    binsaver
    binsaver/ut_util
    binsaver/ut
    cgiparam
    cgiparam/fuzz
    cgiparam/ut
    colorizer
    colorizer/ut
    containers
    coroutine
    cppparser
    cpuid_check
    diff
    diff/ut
    digest
    dot_product
    dot_product/bench
    dot_product/ut
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
    malloc
    lfalloc
    lfalloc/dbg
    lfalloc/dbg_info
    lfalloc/yt
    on_disk
    openssl
    pop_count
    pop_count/benchmark
    pop_count/ut
    sse
    streams
    string_utils
    terminate_handler
    terminate_handler/sample
    text_processing
    threading
    tokenizer
    tokenizer/ut
    unicode
    unittest
    unittest/fat
    unittest/main
    unittest/ut
    yson
    yson/node
    yson/node/pybind
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
    sse/ut
)
ENDIF()
