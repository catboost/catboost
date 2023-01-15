#pragma once

#include "leaf_skipper.h"
#include <cstddef>

class IOutputStream;

namespace NCompactTrie {
    size_t MeasureOffset(size_t offset);

    enum EMinimizeMode {
        MM_DEFAULT, // alollocate new memory for minimized tree
        MM_NOALLOC, // minimize tree in the same buffer
        MM_INPLACE  // do not write tree to the stream, but move to the buffer beginning
    };

    // Return value: size of the minimized trie.
    size_t RawCompactTrieMinimizeImpl(IOutputStream& os, TOpaqueTrie& trie, bool verbose, size_t minMergeSize, EMinimizeMode mode);

    // Return value: size of the minimized trie.
    template <class TPacker>
    size_t CompactTrieMinimizeImpl(IOutputStream& os, const char* data, size_t datalength, bool verbose, const TPacker* packer, EMinimizeMode mode) {
        TPackerLeafSkipper<TPacker> skipper(packer);
        size_t minmerge = MeasureOffset(datalength);
        TOpaqueTrie trie(data, datalength, skipper);
        return RawCompactTrieMinimizeImpl(os, trie, verbose, minmerge, mode);
    }

}
