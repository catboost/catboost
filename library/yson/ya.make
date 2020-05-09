LIBRARY()

# Temporary proxy library until we moved all consumers
# to library/cpp/yson



SRCS(
    all.cpp
)

PEERDIR(library/cpp/yson)

END()
