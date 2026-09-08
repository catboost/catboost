#include "index_reader.h"

#include <util/generic/yexception.h>
#include <util/system/unaligned_mem.h>

#include <climits>

namespace NHnsw {
    namespace {
#pragma pack(push, 1)
        struct TUi32IndexHeader {
            ui32 NumItems;
            ui32 MaxNeighbors;
            ui32 LevelSizeDecay;
        };
#pragma pack(pop)
        static_assert(sizeof(TUi32IndexHeader) == 12);

        /**
         * @brief Whether the blob is a packed graph rather than a ui32 one.
         */
        bool IsPackedIndex(const TBlob& blob) {
            if (blob.Size() < sizeof(TPackedIndexHeader)) {
                return false;
            }
            const ui64 magic = ReadUnaligned<ui64>(blob.Begin());
            return magic == PACKED_INDEX_MAGIC;
        }

        void ReadPackedLayout(const TBlob& blob, THnswIndexLayout* layout) {
            Y_ENSURE(
                IsLittleEndianTarget(),
                "packed hnsw neighbor id formats are implemented for little-endian targets only"
            );

            const TPackedIndexHeader header = ReadUnaligned<TPackedIndexHeader>(blob.Begin());

            Y_ENSURE(
                header.Version == PACKED_INDEX_VERSION,
                "unsupported packed hnsw index version " << header.Version
            );
            const auto format = static_cast<ENeighborIdFormat>(header.Format);
            Y_ENSURE(
                format == ENeighborIdFormat::Ui24 || format == ENeighborIdFormat::BitPacked,
                "unexpected packed hnsw neighbor id format " << header.Format
            );
            Y_ENSURE(
                header.BitsPerId == FormatBitsPerId(format, header.NumItems),
                "packed hnsw bit width " << header.BitsPerId << " does not match its format"
            );
            Y_ENSURE(header.Reserved == 0, "packed hnsw reserved header word is not zero");

            const auto geometry = ComputeLevels(
                format,
                header.NumItems,
                header.MaxNeighbors,
                header.LevelSizeDecay
            );
            Y_ENSURE(
                geometry.size() == header.NumLevels,
                "packed hnsw level geometry does not match its header"
            );
            const size_t expectedSize = sizeof(TPackedIndexHeader)
                + PackedPayloadBits(geometry, header.BitsPerId) / CHAR_BIT
                + PACKED_INDEX_TAIL_GUARD_BYTES;
            Y_ENSURE(
                blob.Size() == expectedSize,
                "packed hnsw size does not match its level geometry"
            );

            layout->Format = format;
            layout->BitsPerId = header.BitsPerId;
            layout->Payload = reinterpret_cast<const ui8*>(blob.Begin()) + sizeof(TPackedIndexHeader);
            layout->Levels = ToLevelLayouts(geometry);
        }

        void ReadUi32Layout(const TBlob& blob, THnswIndexLayout* layout) {
            Y_ENSURE(blob.Size() >= sizeof(TUi32IndexHeader), "hnsw index blob is too short to hold a header");
            const TUi32IndexHeader header = ReadUnaligned<TUi32IndexHeader>(blob.Begin());
            Y_ENSURE(header.LevelSizeDecay > 1, "levelSizeDecay should be greater than 1");

            const auto geometry = ComputeLevels(
                ENeighborIdFormat::Ui32,
                header.NumItems,
                header.MaxNeighbors,
                header.LevelSizeDecay
            );
            Y_ENSURE(
                blob.Size() == sizeof(TUi32IndexHeader) + PayloadBits(geometry, 32) / CHAR_BIT,
                "hnsw index size does not match its level geometry"
            );

            layout->Payload = reinterpret_cast<const ui8*>(blob.Begin()) + sizeof(TUi32IndexHeader);
            layout->Levels = ToLevelLayouts(geometry);
        }
    }

    void THnswIndexReader::ReadIndex(const TBlob& blob, THnswIndexLayout* layout) const {
        *layout = {};
        if (blob.Empty()) {
            return;
        }
        if (IsPackedIndex(blob)) {
            ReadPackedLayout(blob, layout);
        } else {
            ReadUi32Layout(blob, layout);
        }
    }

    void THnswIndexReader::ReadIndex(
        const TBlob& blob,
        TVector<ui32>* numNeighborsInLevels,
        TVector<const ui32*>* levels
    ) const {
        if (blob.Empty()) {
            return;
        }

        THnswIndexLayout layout;
        ReadIndex(blob, &layout);
        Y_ENSURE(
            layout.Format == ENeighborIdFormat::Ui32,
            "packed hnsw neighbor ids require the layout-aware ReadIndex overload"
        );

        for (const auto& level : layout.Levels) {
            levels->push_back(reinterpret_cast<const ui32*>(layout.Payload + (level.BitOffset >> 3)));
            numNeighborsInLevels->push_back(level.NumNeighbors);
        }
    }
} // namespace Hnsw
