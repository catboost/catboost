LIBRARY()



GENERATE_ENUM_SERIALIZATION(priority.h)

SRCS(
    log.cpp
    system.cpp
    file.cpp
    rotating_file.cpp
    null.cpp
    backend.cpp
    thread.cpp
    stream.cpp
    sync_page_cache_file.cpp
    element.cpp
    all.h
    priority.h
    record.h
    filter.cpp
)

END()
