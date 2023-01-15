

LIBRARY()

SRCS(
    bpe_builder.cpp
    bpe_dictionary.cpp
    bpe_helpers.cpp
    dictionary.cpp
    dictionary_builder.cpp
    fbs_helpers.cpp
    frequency_based_dictionary.cpp
    frequency_based_dictionary_impl.cpp
    mmap_frequency_based_dictionary.cpp
    mmap_frequency_based_dictionary_impl.cpp
    mmap_hash_table.cpp
    multigram_dictionary_helpers.cpp
    options.cpp
    serialization_helpers.cpp
    util.cpp
)

PEERDIR(
    library/cpp/containers/flat_hash
    library/cpp/json
    library/cpp/text_processing/dictionary/idl
    library/cpp/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(types.h)

END()

RECURSE(
    idl
    ut
)
