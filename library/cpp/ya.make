

RECURSE(
    accurate_accumulate
    accurate_accumulate/benchmark
    accurate_accumulate/benchmark/metrics
    archive/ut
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
    float16
    float16/ut
    grid_creator
    grid_creator/fuzz
    grid_creator/ut
    malloc
    on_disk
    pop_count
    pop_count/benchmark
    pop_count/ut
    streams
    string_utils
    terminate_handler
    terminate_handler/sample
    tokenizer
    tokenizer/ut
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
    
)
ENDIF()
