#pragma once

#include <util/memory/blob.h>

namespace NHnsw {
    struct THnswIndexLayout;

    /** @brief Reads the layout of a graph stored in either built-in HNSW format. */
    class THnswIndexReader {
    public:
        void ReadIndex(const TBlob& blob, THnswIndexLayout* layout) const;
    };
} // namespace NHnsw
