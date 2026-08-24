#pragma once

#include <library/cpp/hnsw/helpers/neighbor_id_format.h>

#include <util/memory/blob.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <climits>

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

    /**
     * @brief Layout of an index whose reader only reports raw ui32 level pointers.
     *
     * Such a reader cannot be describing anything but the ui32 layout, so its output determines
     * the whole layout. Kept so that a custom index reader implementing just that older interface
     * keeps working unchanged.
     *
     * The offsets are taken from the lowest reported pointer rather than from the first one, so a
     * reader is free to hand out its levels in any order and out of any buffers: @p Payload plus
     * the stored offset reproduces each reported pointer exactly.
     */
    inline THnswIndexLayout MakeUi32Layout(
        const TVector<ui32>& numNeighborsInLevels,
        const TVector<const ui32*>& levels
    ) {
        Y_ENSURE(
            numNeighborsInLevels.size() == levels.size(),
            "index reader reported " << levels.size() << " levels but "
            << numNeighborsInLevels.size() << " neighbor counts"
        );
        THnswIndexLayout layout;
        if (levels.empty()) {
            return layout;
        }
        layout.Payload = reinterpret_cast<const ui8*>(levels[0]);
        for (const ui32* level : levels) {
            layout.Payload = Min(layout.Payload, reinterpret_cast<const ui8*>(level));
        }
        layout.Levels.reserve(levels.size());
        for (size_t level = 0; level < levels.size(); ++level) {
            layout.Levels.push_back({
                .NumNeighbors = numNeighborsInLevels[level],
                .BitOffset = ui64(reinterpret_cast<const ui8*>(levels[level]) - layout.Payload) * CHAR_BIT,
            });
        }
        return layout;
    }

    class THnswIndexReader {
    public:
        void ReadIndex(const TBlob& blob, TVector<ui32>* numNeighborsInLevels, TVector<const ui32*>* levels) const;
        void ReadIndex(const TBlob& blob, THnswIndexLayout* layout) const;
    };
} // namespace Hnsw
