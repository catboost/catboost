#pragma once

#include <util/memory/blob.h>

namespace NHnsw {
    struct THnswIndexLayout;
} // namespace NHnsw

namespace NOnlineHnsw {
    class TOnlineHnswIndexReader {
    public:
        void ReadIndex(const TBlob& blob, NHnsw::THnswIndexLayout* layout) const;
    };
} // namespace NOnlineHnsw
