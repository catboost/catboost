#pragma once

#include <util/generic/fwd.h>
#include <stddef.h>

class IOutputStream;

namespace NHnsw {
    struct THnswIndexData;

    /**
 * @brief Method for writing freshly built HNSW indexes.
 *
 * Typical usage is as follows:
 * @code
 *   THnswIndexData indexData = BuildIndex<TDistance>(opts, itemStorage);
 *   WriteIndex(indexData, out);
 * @endcode
 *
 * Please, refer to hnsw/ut/main.cpp for a comprehensive usage example.
 */
    size_t ExpectedSize(const THnswIndexData& index);
    void WriteIndex(const THnswIndexData& index, IOutputStream& out);
    void WriteIndex(const THnswIndexData& index, const TString& outputFilename);

    void DebugIndexDump(const THnswIndexData& index, IOutputStream& out);

}
