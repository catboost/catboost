LIBRARY()



SRCS(
    array_with_size.h
    chunked_helpers_trie.h
    comptrie_builder.h
    comptrie.h
    comptrie_impl.h
    comptrie_packer.h
    comptrie_trie.h
    first_symbol_iterator.h
    key_selector.h
    leaf_skipper.h
    make_fast_layout.h
    minimize.h
    node.h
    opaque_trie_iterator.h
    pattern_searcher.h
    prefix_iterator.h
    protopacker.h
    search_iterator.h
    set.h
    writeable_node.h
    write_trie_backwards.h
)

PEERDIR(
    library/cpp/containers/comptrie
)

END()
