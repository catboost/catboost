UNITTEST()



SRCS(
    containers_ut.cpp
    flat_hash_ut.cpp
    iterator_ut.cpp
    probings_ut.cpp
    size_fitters_ut.cpp
    table_ut.cpp
)

PEERDIR(
    library/cpp/containers/flat_hash
)

END()
