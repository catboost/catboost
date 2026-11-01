#pragma once

#include <library/cpp/hnsw/helpers/neighbor_id_format.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NHnsw {
    /**
     * @brief Everything the search loop needs to address a neighbor row: the id width, and per
     * level the offset of its first row measured in bits from @p Payload.
     */
    struct THnswIndexLayout {
        ENeighborIdFormat Format = ENeighborIdFormat::Ui32;
        ui32 BitsPerId = 32;
        const ui8* Payload = nullptr;
        TVector<TNeighborLevelLayout> Levels;
    };
} // namespace NHnsw
