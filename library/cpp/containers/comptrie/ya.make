LIBRARY()



SRCS(
    array_with_size.h
    chunked_helpers_trie.h
    comptrie.h
    comptrie_packer.h
    comptrie_trie.h
    first_symbol_iterator.h
    key_selector.h
    leaf_skipper.h
    set.h
    comptrie.cpp
    comptrie_builder.cpp
    comptrie_impl.cpp
    make_fast_layout.cpp
    minimize.cpp
    node.cpp
    opaque_trie_iterator.cpp
    prefix_iterator.cpp
    search_iterator.cpp
    write_trie_backwards.cpp
    writeable_node.cpp
)

PEERDIR(
    library/packers
    library/cpp/containers/compact_vector
    library/cpp/on_disk/chunks
    util/draft
)

END()
