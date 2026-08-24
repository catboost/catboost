#include "index_writer.h"
#include "index_data.h"

#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/stream/file.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/system/unaligned_mem.h>

#include <climits>
#include <cstring>
#include <numeric>


namespace NHnsw {
    namespace {
        /**
         * @brief Writes @p count ids, @p bitsPerId bits each, into @p dst.
         *
         * Every id is OR'd into the 64-bit window holding it, so @p dst must be zeroed beforehand
         * and must have PACKED_INDEX_TAIL_GUARD_BYTES of slack past the last id. Ui24 needs no
         * path of its own: at a width that is a whole number of bytes every shift is zero, so this
         * loop emits exactly the three bytes per id a memcpy would.
         */
        void PackIds(ui8* dst, ui32 bitsPerId, const ui32* ids, ui64 count) {
            for (ui64 i = 0, bitOffset = 0; i < count; ++i, bitOffset += bitsPerId) {
                ui8* const at = dst + (bitOffset >> 3);
                auto window = ReadUnaligned<ui64>(at);
                window |= ui64(ids[i]) << (bitOffset & 7);
                WriteUnaligned<ui64>(at, window);
            }
        }

        /**
         * @brief How many rows to pack at a time, keeping the writer's transient buffer bounded.
         *
         * A chunk must end on a 64-bit boundary so that no id straddles two chunks and so that the
         * level stays aligned exactly as ComputeLevels describes it.
         */
        ui64 RowsPerChunk(ui64 rowBits, ui64 numRows) {
            const ui64 rowsPerAlignedGroup = 64 / std::gcd(rowBits, ui64(64));
            const ui64 targetBits = 1 << 19;
            const ui64 groups = Max<ui64>(1, targetBits / (rowsPerAlignedGroup * rowBits));
            return Min<ui64>(numRows, rowsPerAlignedGroup * groups);
        }

        void WritePackedIndex(const THnswIndexData& index, IOutputStream& out, ENeighborIdFormat format) {
            Y_ENSURE(
                IsLittleEndianTarget(),
                "packed hnsw neighbor id formats are implemented for little-endian targets only"
            );

            const ui32 bitsPerId = FormatBitsPerId(format, index.NumItems);
            const auto levels = ComputeLevels(
                format,
                index.NumItems,
                index.MaxNeighbors,
                index.LevelSizeDecay
            );

            Y_ENSURE(
                bitsPerId <= 32,
                "packed hnsw neighbor ids are at most 32 bits wide, but format "
                << format << " asks for " << bitsPerId
            );
            const ui64 idLimit = bitsPerId == 32 ? Max<ui32>() : (ui64(1) << bitsPerId) - 1;
            Y_ENSURE(
                index.NumItems == 0 || index.NumItems - 1 <= idLimit,
                "a graph of " << index.NumItems << " items does not fit into " << bitsPerId << " bit ids"
            );
            const auto* const maxId = MaxElement(index.FlatLevels.begin(), index.FlatLevels.end());
            Y_ENSURE(
                maxId == index.FlatLevels.end() || *maxId <= idLimit,
                "neighbor id " << *maxId << " does not fit into " << bitsPerId << " bits"
            );

            ui64 numIds = 0;
            for (const auto& level : levels) {
                numIds += ui64(level.NumItems) * level.NumNeighbors;
            }
            Y_ENSURE(
                numIds == index.FlatLevels.size(),
                "level geometry describes " << numIds << " neighbor ids but FlatLevels holds "
                << index.FlatLevels.size()
            );

            const TPackedIndexHeader header{
                .Magic = PACKED_INDEX_MAGIC,
                .Version = PACKED_INDEX_VERSION,
                .Format = static_cast<ui32>(format),
                .BitsPerId = bitsPerId,
                .NumItems = index.NumItems,
                .MaxNeighbors = index.MaxNeighbors,
                .LevelSizeDecay = index.LevelSizeDecay,
                .NumLevels = static_cast<ui32>(levels.size()),
            };
            out.Write(&header, sizeof(header));

            TBuffer chunk;
            ui64 pos = 0;
            ui64 payloadBytes = 0;
            for (const auto& level : levels) {
                const ui64 rowBits = ui64(level.NumNeighbors) * bitsPerId;
                if (rowBits == 0) {
                    continue;
                }
                const ui64 rowsPerChunk = RowsPerChunk(rowBits, level.NumItems);
                chunk.Resize(AlignUp<ui64>(rowsPerChunk * rowBits, 64) / CHAR_BIT + PACKED_INDEX_TAIL_GUARD_BYTES);
                for (ui64 firstRow = 0; firstRow < level.NumItems; firstRow += rowsPerChunk) {
                    const ui64 rows = Min<ui64>(rowsPerChunk, level.NumItems - firstRow);
                    const ui64 count = rows * level.NumNeighbors;
                    memset(chunk.Data(), 0, chunk.Size());
                    PackIds(
                        reinterpret_cast<ui8*>(chunk.Data()),
                        bitsPerId,
                        index.FlatLevels.data() + pos,
                        count
                    );
                    const ui64 chunkBytes = AlignUp<ui64>(rows * rowBits, 64) / CHAR_BIT;
                    out.Write(chunk.Data(), chunkBytes);
                    pos += count;
                    payloadBytes += chunkBytes;
                }
            }
            Y_ENSURE(
                pos == index.FlatLevels.size(),
                "packed hnsw wrote " << pos << " of " << index.FlatLevels.size() << " neighbor ids"
            );
            Y_ENSURE(
                payloadBytes == PackedPayloadBits(levels, bitsPerId) / CHAR_BIT,
                "packed hnsw wrote " << payloadBytes << " payload bytes but its level geometry describes "
                << PackedPayloadBits(levels, bitsPerId) / CHAR_BIT
            );

            const ui64 tailGuard = 0;
            static_assert(sizeof(tailGuard) == PACKED_INDEX_TAIL_GUARD_BYTES);
            out.Write(&tailGuard, sizeof(tailGuard));
        }
    }

    void DebugIndexDump(const THnswIndexData& index, IOutputStream& out) {
        out << "Header:"
            << " NumItems=" << index.NumItems
            << " MaxNeighbors=" << index.MaxNeighbors
            << " LevelSizeDecay=" << index.LevelSizeDecay
            << "\n";

        out << "Items dump: \n\n";

        TVector<const ui32*> levels;
        TVector<ui32> numNeighborsInLevels;
        TVector<ui32> numItemsInLevels;
        {
            const ui32* data = index.FlatLevels.begin();
            for (i64 numItems = index.NumItems; numItems > 1; numItems /= index.LevelSizeDecay) {
                Y_ENSURE(data < index.FlatLevels.end());
                levels.push_back(data);
                numNeighborsInLevels.push_back(Min<i64>(index.MaxNeighbors, numItems - 1));
                numItemsInLevels.push_back(numItems);
                data += numItems * numNeighborsInLevels.back();
            }

            Y_ENSURE(data == index.FlatLevels.end());
        }

        for (auto levelNum : xrange<i64>(levels.size() - 1, -1, -1)) {
            for (auto itemId : xrange(numItemsInLevels[levelNum])) {
                out << "At level " << levelNum << " of id " << itemId << ":";
                for (auto neighborId : xrange(numNeighborsInLevels[levelNum])) {
                    out << " " << (levels[levelNum] + itemId * numNeighborsInLevels[levelNum])[neighborId];
                }
                out << "\n";
            }
        }
    }

    size_t ExpectedSize(const THnswIndexData& index, ENeighborIdFormat format) {
        if (format == ENeighborIdFormat::Ui32) {
            return sizeof(index.NumItems)
                + sizeof(index.MaxNeighbors)
                + sizeof(index.LevelSizeDecay)
                + index.FlatLevels.size() * sizeof(index.FlatLevels[0]);
        }
        const ui32 bitsPerId = FormatBitsPerId(format, index.NumItems);
        const auto levels = ComputeLevels(
            format,
            index.NumItems,
            index.MaxNeighbors,
            index.LevelSizeDecay
        );
        return sizeof(TPackedIndexHeader)
            + PackedPayloadBits(levels, bitsPerId) / CHAR_BIT
            + PACKED_INDEX_TAIL_GUARD_BYTES;
    }

    void WriteIndex(const THnswIndexData& index, IOutputStream& out, ENeighborIdFormat format) {
        if (format != ENeighborIdFormat::Ui32) {
            WritePackedIndex(index, out, format);
            return;
        }
        out.Write(&index.NumItems, sizeof(index.NumItems));
        out.Write(&index.MaxNeighbors, sizeof(index.MaxNeighbors));
        out.Write(&index.LevelSizeDecay, sizeof(index.LevelSizeDecay));
        out.Write(index.FlatLevels.data(), index.FlatLevels.size() * sizeof(index.FlatLevels[0]));
    }

    void WriteIndex(const THnswIndexData& index, const TString& outputFilename, ENeighborIdFormat format) {
        TFixedBufferFileOutput out(outputFilename);
        WriteIndex(index, out, format);
    }

}
