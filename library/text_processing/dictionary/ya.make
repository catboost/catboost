

LIBRARY()

SRCS(
    app_helpers.cpp
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
    library/containers/flat_hash
    library/json
    library/text_processing/dictionary/idl
    library/text_processing/tokenizer
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(types.h)

END()

RECURSE(
    idl
    ut
)
