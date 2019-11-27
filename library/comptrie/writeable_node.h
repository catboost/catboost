#pragma once

#include <cstddef>

namespace NCompactTrie {
    class TNode;

    class TWriteableNode {
    public:
        const char* LeafPos;
        size_t LeafLength;

        size_t ForwardOffset;
        size_t LeftOffset;
        size_t RightOffset;
        char Label;

        TWriteableNode();
        TWriteableNode(const TNode& node, const char* data);

        // When you call this, the offsets should be relative to the end of the node. Use NPOS to indicate an absent offset.
        size_t Pack(char* buffer) const;
        size_t Measure() const;
    };

}
