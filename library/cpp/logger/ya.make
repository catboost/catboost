

LIBRARY()

GENERATE_ENUM_SERIALIZATION(priority.h)

PEERDIR (
    library/cpp/json
)

SRCS(
    all.h
    backend.cpp
    backend_creator.cpp
    composite.cpp
    GLOBAL composite_creator.cpp
    element.cpp
    file.cpp
    GLOBAL file_creator.cpp
    filter.cpp
    filter_creator.cpp
    log.cpp
    null.cpp
    GLOBAL null_creator.cpp
    priority.h
    record.h
    rotating_file.cpp
    GLOBAL rotating_file_creator.cpp
    stream.cpp
    GLOBAL stream_creator.cpp
    sync_page_cache_file.cpp
    GLOBAL sync_page_cache_file_creator.cpp
    system.cpp
    GLOBAL system_creator.cpp
    thread.cpp
    thread_creator.cpp
    GLOBAL uninitialized_creator.cpp
)

END()

RECURSE_FOR_TESTS(ut)
