LIBRARY()

LICENSE(Apache-2.0)



NO_UTIL()
NO_COMPILER_WARNINGS()

# https://github.com/google/tcmalloc
VERSION(2020-11-23-a643d89610317be1eff9f7298104eef4c987d8d5)

SRCS(
    tcmalloc/arena.cc
    tcmalloc/background.cc
    tcmalloc/central_freelist.cc
    tcmalloc/common.cc
    tcmalloc/cpu_cache.cc
    tcmalloc/experimental_56_size_class.cc
    tcmalloc/experiment.cc
    tcmalloc/guarded_page_allocator.cc
    tcmalloc/huge_address_map.cc
    tcmalloc/huge_allocator.cc
    tcmalloc/huge_cache.cc
    tcmalloc/huge_page_aware_allocator.cc
    tcmalloc/internal/environment.cc
    tcmalloc/internal/logging.cc
    tcmalloc/internal/memory_stats.cc
    tcmalloc/internal/mincore.cc
    tcmalloc/internal/percpu.cc
    tcmalloc/internal/percpu_rseq_asm.S
    tcmalloc/internal/percpu_rseq_unsupported.cc
    tcmalloc/internal/util.cc
    tcmalloc/legacy_size_classes.cc
    tcmalloc/noruntime_size_classes.cc
    tcmalloc/page_allocator.cc
    tcmalloc/page_allocator_interface.cc
    tcmalloc/page_heap.cc
    tcmalloc/pagemap.cc
    tcmalloc/parameters.cc
    tcmalloc/peak_heap_tracker.cc
    tcmalloc/sampler.cc
    tcmalloc/size_classes.cc
    tcmalloc/span.cc
    tcmalloc/stack_trace_table.cc
    tcmalloc/static_vars.cc
    tcmalloc/stats.cc
    tcmalloc/system-alloc.cc
    tcmalloc/tcmalloc.cc
    tcmalloc/thread_cache.cc
    tcmalloc/transfer_cache.cc
)

PEERDIR(
    contrib/restricted/abseil-cpp
    contrib/libs/tcmalloc/malloc_extension
)

CFLAGS(-DTCMALLOC_256K_PAGES)

END()
