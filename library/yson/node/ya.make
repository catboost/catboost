LIBRARY()

# Temporary proxy library until we moved all consumers
# to library/cpp/yson/node



SRCS(
    all.cpp
)

PEERDIR(
    library/cpp/yson/node
)

END()
