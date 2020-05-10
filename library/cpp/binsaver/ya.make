LIBRARY()



SRCS(
    class_factory.h
    bin_saver.cpp
    blob_io.cpp
    buffered_io.cpp
    mem_io.cpp
    util_stream_io.cpp
)

PEERDIR(
    library/cpp/containers/2d_array
)

END()
