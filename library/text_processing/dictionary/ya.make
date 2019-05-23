

LIBRARY()

SRCS(
    options.cpp
    dictionary.cpp
    dictionary_builder.cpp
    bpe_dictionary.cpp
    frequency_based_dictionary.cpp
    frequency_based_dictionary_impl.cpp
    bpe_builder.cpp
    util.cpp
    bpe_builder.cpp
)

PEERDIR(
    library/containers/flat_hash
    library/json
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(types.h)

END()
