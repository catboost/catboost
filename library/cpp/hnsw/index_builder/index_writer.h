#pragma once

#include <library/cpp/hnsw/helpers/neighbor_id_format.h>

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
 *
 * The format argument picks the on-disk width of a neighbor id:
 *   - ENeighborIdFormat::Ui32, the default, emits the bytes this library has always written;
 *   - ENeighborIdFormat::Ui24 and ENeighborIdFormat::BitPacked store ids at a narrower width.
 * All of them are lossless, but the two packed ones produce a blob that only a reader knowing the
 * packed layout can parse — THnswIndexReader does, an older reader fails loudly on it.
 */
    size_t ExpectedSize(const THnswIndexData& index, ENeighborIdFormat format = ENeighborIdFormat::Ui32);
    void WriteIndex(
        const THnswIndexData& index,
        IOutputStream& out,
        ENeighborIdFormat format = ENeighborIdFormat::Ui32
    );
    void WriteIndex(
        const THnswIndexData& index,
        const TString& outputFilename,
        ENeighborIdFormat format = ENeighborIdFormat::Ui32
    );

    void DebugIndexDump(const THnswIndexData& index, IOutputStream& out);

}
