#pragma once

#include "minimize.h"

#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <cstddef>

namespace NCompactTrie {
    class TReverseNodeEnumerator {
    public:
        virtual ~TReverseNodeEnumerator() = default;
        virtual bool Move() = 0;
        virtual size_t GetLeafLength() const = 0;
        virtual size_t RecreateNode(char* buffer, size_t resultLength) = 0;
    };

    struct TOpaqueTrie;

    size_t WriteTrieBackwards(IOutputStream& os, TReverseNodeEnumerator& enumerator, bool verbose);
    size_t WriteTrieBackwardsNoAlloc(IOutputStream& os, TReverseNodeEnumerator& enumerator, TOpaqueTrie& trie, EMinimizeMode mode);

}
