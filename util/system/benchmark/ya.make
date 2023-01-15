

RECURSE(
    cpu_id
    cpu_id/metrics
    create_destroy_thread
    create_destroy_thread/metrics
)

IF (NOT OS_WINDOWS)
    RECURSE(
    rdtsc
)
ENDIF()
