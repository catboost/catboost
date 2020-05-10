LIBRARY()



PEERDIR(
    library/cpp/containers/stack_vector
    library/cpp/json/writer
    library/yson
)

SRCS(
    blocking_queue.cpp
    consumer.cpp
    event.cpp
    gettime.cpp
    global.cpp
    guard.cpp
    json.cpp
    queue.cpp
    sampler.cpp
    samplers.cpp
    sync.cpp
    tracer.cpp
    yson.cpp
    counter.cpp
    interface.cpp
    sample_value.cpp
    saveload.cpp
)

END()
