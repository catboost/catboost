#include "neighbor_id_format.h"

#include <util/generic/utility.h>
#include <util/system/align.h>

#include <bit>

namespace NHnsw {
    TVector<TNeighborLevelLayout> ToLevelLayouts(const TVector<TNeighborLevelGeometry>& geometry) {
        TVector<TNeighborLevelLayout> levels;
        levels.reserve(geometry.size());
        for (const auto& level : geometry) {
            levels.push_back({.NumNeighbors = level.NumNeighbors, .BitOffset = level.BitOffset});
        }
        return levels;
    }

    ui32 BitsForNumItems(ui32 numItems) {
        return static_cast<ui32>(std::bit_width(numItems > 1 ? numItems - 1 : ui32(1)));
    }

    ui32 FormatBitsPerId(ENeighborIdFormat format, ui32 numItems) {
        switch (format) {
            case ENeighborIdFormat::Ui32:
                return 32;
            case ENeighborIdFormat::Ui24:
                return 24;
            case ENeighborIdFormat::BitPacked:
                return BitsForNumItems(numItems);
        }
        ythrow yexception() << "unknown neighbor id format " << static_cast<ui32>(format);
    }

    ui64 LevelBits(const TNeighborLevelGeometry& level, ui32 bitsPerId) {
        const ui64 rowBits = ui64(level.NumNeighbors) * bitsPerId;
        Y_ENSURE(
            rowBits == 0 || level.NumItems <= PACKED_INDEX_MAX_PAYLOAD_BITS / rowBits,
            "hnsw level geometry does not fit into " << PACKED_INDEX_MAX_PAYLOAD_BITS << " bits"
        );
        const ui64 levelBits = ui64(level.NumItems) * rowBits;
        Y_ENSURE(
            level.BitOffset <= PACKED_INDEX_MAX_PAYLOAD_BITS - levelBits,
            "hnsw level geometry does not fit into " << PACKED_INDEX_MAX_PAYLOAD_BITS << " bits"
        );
        return levelBits;
    }

    TVector<TNeighborLevelGeometry> ComputeLevels(
        ENeighborIdFormat format,
        ui32 numItems,
        ui32 maxNeighbors,
        ui32 levelSizeDecay
    ) {
        TVector<TNeighborLevelGeometry> levels;
        if (numItems == 0) {
            return levels;
        }
        Y_ENSURE(levelSizeDecay > 1, "levelSizeDecay should be greater than 1");
        if (numItems == 1) {
            levels.push_back({.NumItems = 1, .NumNeighbors = 0, .BitOffset = 0});
            return levels;
        }
        const ui32 bitsPerId = FormatBitsPerId(format, numItems);
        const bool alignLevels = format != ENeighborIdFormat::Ui32;
        ui64 bitOffset = 0;
        for (ui32 items = numItems; items > 1; items /= levelSizeDecay) {
            const ui32 numNeighbors = Min(maxNeighbors, items - 1);
            levels.push_back({.NumItems = items, .NumNeighbors = numNeighbors, .BitOffset = bitOffset});
            bitOffset += LevelBits(levels.back(), bitsPerId);
            if (alignLevels) {
                bitOffset = AlignUp<ui64>(bitOffset, 64);
            }
        }
        return levels;
    }

    ui64 PayloadBits(const TVector<TNeighborLevelGeometry>& levels, ui32 bitsPerId) {
        if (levels.empty()) {
            return 0;
        }
        const auto& last = levels.back();
        return last.BitOffset + LevelBits(last, bitsPerId);
    }

    ui64 PackedPayloadBits(const TVector<TNeighborLevelGeometry>& levels, ui32 bitsPerId) {
        return AlignUp<ui64>(PayloadBits(levels, bitsPerId), 64);
    }

} // namespace NHnsw
