#pragma once

#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/compiler.h>
#include <util/system/types.h>
#include <util/system/unaligned_mem.h>

#include <bit>
#include <stddef.h>

namespace NHnsw {
    /**
     * @brief Storage width of a neighbor id inside a serialized hnsw graph.
     *
     * Ui32 is the format that has always been written: a 12-byte header (NumItems, MaxNeighbors,
     * LevelSizeDecay) followed by a flat ui32 array holding, level by level, one row of
     * NumNeighbors ids per item. It carries no marker of its own.
     *
     * Ui24 and BitPacked prefix the graph with TPackedIndexHeader, whose first eight bytes are
     * PACKED_INDEX_MAGIC, so a reader can tell the layouts apart and an older reader fails loudly
     * instead of misparsing. What follows the header is the same sequence of levels and rows, with
     * every id stored in TPackedIndexHeader::BitsPerId bits instead of 32: 24 bits for Ui24, and
     * for BitPacked as many as the item count needs, which is what BitsForNumItems computes. Ids
     * follow one another with no padding, least significant bit first, so a row of N ids occupies
     * N * BitsPerId bits; only level starts are padded, each rounded up to a 64-bit boundary. The
     * payload is followed by PACKED_INDEX_TAIL_GUARD_BYTES zero bytes, so the unpacker may always
     * read a whole 64-bit window at the last id.
     *
     * No level geometry is stored in either layout: ComputeLevels rederives it from NumItems,
     * MaxNeighbors and LevelSizeDecay, exactly as a ui32 reader has always done.
     */
    enum class ENeighborIdFormat : ui32 {
        Ui32 = 0,
        Ui24 = 1,
        BitPacked = 2,
    };

    /**
     * @brief Calls @p func with the neighbor id format as a compile-time parameter.
     *
     * The one place the enum is walked: everything that picks code per format goes through here,
     * so a fourth format means touching this function and the things it dispatches to, rather than
     * a switch repeated at every site.
     */
    template <class TFunc>
    Y_FORCE_INLINE decltype(auto) DispatchByFormat(ENeighborIdFormat format, TFunc&& func) {
        switch (format) {
            case ENeighborIdFormat::Ui32:
                return func.template operator()<ENeighborIdFormat::Ui32>();
            case ENeighborIdFormat::Ui24:
                return func.template operator()<ENeighborIdFormat::Ui24>();
            case ENeighborIdFormat::BitPacked:
                return func.template operator()<ENeighborIdFormat::BitPacked>();
        }
        ythrow yexception() << "unknown neighbor id format " << static_cast<ui32>(format);
    }

    /**
     * @brief "HNSWPACK" in little-endian byte order.
     *
     * Eight bytes wide rather than four because a ui32 graph carries no marker of its own: its
     * first words are NumItems and MaxNeighbors, so a four-byte magic is matched by any graph of
     * exactly that many items. Spanning both words means a collision would need MaxNeighbors to be
     * 0x4B434150, over a billion, which no graph degree ever reaches.
     */
    constexpr ui64 PACKED_INDEX_MAGIC = 0x4B43415057534E48ULL;
    constexpr ui32 PACKED_INDEX_VERSION = 1;
    constexpr size_t PACKED_INDEX_TAIL_GUARD_BYTES = 8;
    constexpr ui64 PACKED_INDEX_MAX_PAYLOAD_BITS = ui64(1) << 60;

    constexpr bool IsLittleEndianTarget() {
        return std::endian::native == std::endian::little;
    }

    /**
     * @brief What a graph written in a packed neighbor id format starts with.
     *
     * Forty bytes, a multiple of eight, so that the payload following it starts at a 64-bit
     * boundary just as every level inside the payload does.
     */
#pragma pack(push, 1)
    struct TPackedIndexHeader {
        /// Always PACKED_INDEX_MAGIC, which is what tells a packed graph from a ui32 one.
        ui64 Magic;

        /**
         * @brief Ordinal version of the packed layout, currently PACKED_INDEX_VERSION.
         *
         * Bumped whenever the bytes after this header change meaning; a reader refuses every
         * value it was not built for rather than guessing.
         */
        ui32 Version;

        /// The ENeighborIdFormat the ids are stored in: Ui24 or BitPacked, never Ui32.
        ui32 Format;

        /**
         * @brief Width of one stored id, in bits. Between 1 and 32.
         *
         * Always FormatBitsPerId(Format, NumItems): 24 for Ui24, BitsForNumItems(NumItems) for
         * BitPacked. A reader recomputes it and refuses a header that disagrees, so this is a
         * cross-check on the header rather than a width a writer may pick freely.
         */
        ui32 BitsPerId;

        /// Number of items in the graph, which is the row count of its lowest level.
        ui32 NumItems;

        /// Upper bound on a row: a level of @p items items stores Min(MaxNeighbors, items - 1) ids per row.
        ui32 MaxNeighbors;

        /// Ratio between the item counts of two consecutive levels. Greater than 1.
        ui32 LevelSizeDecay;

        /// Number of levels, as ComputeLevels derives them from the three fields above. A cross-check, like BitsPerId.
        ui32 NumLevels;

        /**
         * @brief Zero, and refused by a reader if it is not.
         *
         * Present to round the header up to a multiple of eight bytes, so that the payload it
         * precedes starts at a 64-bit boundary.
         */
        ui32 Reserved = 0;
    };
#pragma pack(pop)
    static_assert(sizeof(TPackedIndexHeader) == 40);

    /**
     * @brief Everything needed to address the rows of one level: the row width and the offset of
     * the level's first row, measured in bits from the start of the payload.
     */
    struct TNeighborLevelLayout {
        ui32 NumNeighbors = 0;
        ui64 BitOffset = 0;
    };

    /**
     * @brief The shape of one level as derived from a graph header: its row count on top of the
     * addressing a reader needs.
     *
     * A reader that only reports level pointers cannot know the row counts, which is why they are
     * kept out of TNeighborLevelLayout instead of being left unfilled in it.
     */
    struct TNeighborLevelGeometry {
        ui32 NumItems = 0;
        ui32 NumNeighbors = 0;
        ui64 BitOffset = 0;
    };

    TVector<TNeighborLevelLayout> ToLevelLayouts(const TVector<TNeighborLevelGeometry>& geometry);

    /**
     * @brief Bits needed to hold any id of a graph of @p numItems items. Never zero, never over 32.
     */
    ui32 BitsForNumItems(ui32 numItems);

    ui32 FormatBitsPerId(ENeighborIdFormat format, ui32 numItems);

    /**
     * @brief Size of one level in bits.
     *
     * Refuses any geometry whose products would not stay inside PACKED_INDEX_MAX_PAYLOAD_BITS. A
     * header that describes more than that is rejected here instead of wrapping around into a
     * small number that would pass a reader's size check.
     */
    ui64 LevelBits(const TNeighborLevelGeometry& level, ui32 bitsPerId);

    /**
     * @brief Level geometry of a graph stored in @p format, measured in bits.
     *
     * The levels are the same ones the ui32 layout has always had. A ui32 graph's levels follow
     * each other with no padding; a packed graph rounds every level start up to a 64-bit boundary,
     * which is what lets a row be unpacked by whole 64-bit windows.
     */
    TVector<TNeighborLevelGeometry> ComputeLevels(
        ENeighborIdFormat format,
        ui32 numItems,
        ui32 maxNeighbors,
        ui32 levelSizeDecay
    );

    ui64 PayloadBits(const TVector<TNeighborLevelGeometry>& levels, ui32 bitsPerId);

    ui64 PackedPayloadBits(const TVector<TNeighborLevelGeometry>& levels, ui32 bitsPerId);

    /**
     * @brief Reads the ids of one packed row, id by id.
     *
     * Only the packed formats go through an unpacker; a ui32 row is handed out as a view into the
     * blob instead.
     */
    template <ENeighborIdFormat IdFormat>
    struct TNeighborRowUnpacker;

    template <>
    struct TNeighborRowUnpacker<ENeighborIdFormat::Ui24> {
        template <class TFunc>
        Y_FORCE_INLINE static void ForEach(const ui8* payload, ui64 bitOffset, ui32 count, ui32 /*bitsPerItem*/, TFunc&& func) {
            Y_ENSURE(
                (bitOffset & 7) == 0,
                "a ui24 neighbor row starts at bit offset " << bitOffset << ", which is not a byte boundary"
            );
            const ui8* row = payload + (bitOffset >> 3);
            for (ui32 i = 0; i < count; ++i, row += 3) {
                const auto window = ReadUnaligned<ui32>(row);
                func(window & 0xFFFFFFu);
            }
        }
    };

    template <>
    struct TNeighborRowUnpacker<ENeighborIdFormat::BitPacked> {
        /**
         * @brief Reads each id out of one 64-bit window.
         *
         * An id starts at most 7 bits into its window, so a window covers any width up to 57. That
         * bound is never approached: FormatBitsPerId returns 32, 24, or BitsForNumItems, which caps
         * itself at 32, so the shift and the mask below are always well within a ui64.
         */
        template <class TFunc>
        Y_FORCE_INLINE static void ForEach(const ui8* payload, ui64 bitOffset, ui32 count, ui32 bitsPerItem, TFunc&& func) {
            const ui64 mask = (ui64(1) << bitsPerItem) - 1;
            for (ui32 i = 0; i < count; ++i, bitOffset += bitsPerItem) {
                const auto window = ReadUnaligned<ui64>(payload + (bitOffset >> 3));
                func(static_cast<ui32>((window >> (bitOffset & 7)) & mask));
            }
        }
    };

} // namespace NHnsw
