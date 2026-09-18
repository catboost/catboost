#pragma once

#include <library/cpp/hnsw/index/core/neighbors_getter.h>
#include <library/cpp/hnsw/helpers/neighbor_id_format.h>

#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#include <type_traits>
#include <utility>
#include <stddef.h>

namespace NHnsw {
    /**
     * @brief The neighbor rows of one level, in whichever id format the index was written in.
     *
     * This is the single row-access point of the search: a neighbors getter reads rows through it
     * instead of walking a raw ui32 array, so the same getter works over every format.
     */
    class TLevelRows {
    public:
        TLevelRows() = default;

        TLevelRows(
            ENeighborIdFormat format,
            const ui8* payload,
            const ui32 bitsPerId,
            const TNeighborLevelLayout& level
        )
            : Format(format)
            , Payload(payload)
            , Ui32Base(
                format == ENeighborIdFormat::Ui32
                ? reinterpret_cast<const ui32*>(payload + (level.BitOffset >> 3))
                : nullptr
            )
            , LevelBitOffset(level.BitOffset)
            , RowBitStride(ui64(level.NumNeighbors) * bitsPerId)
            , NumNeighbors(level.NumNeighbors)
            , BitsPerId(bitsPerId)
        {
        }

        TLevelRows(const ui32* level, const ui32 numNeighbors)
            : TLevelRows(
                ENeighborIdFormat::Ui32,
                reinterpret_cast<const ui8*>(level),
                32,
                TNeighborLevelLayout{.NumNeighbors = numNeighbors, .BitOffset = 0}
            )
        {
        }

        ui32 GetNumNeighbors() const {
            return NumNeighbors;
        }

        ENeighborIdFormat GetFormat() const {
            return Format;
        }

        /**
         * @brief Walks row @p id, unpacking it as @p IdFormat.
         *
         * @p IdFormat is a template parameter so that the loop holds one format's unpacking and
         * @p func is inlined into it, rather than reached through an indirection once per id.
         */
        template <ENeighborIdFormat IdFormat, CNeighborVisitor TFunc>
        Y_FORCE_INLINE void ForEachInRowAs(const ui32 id, TFunc&& func) const {
            // A local, so that a func writing to memory cannot force a re-read of the row width.
            const ui32 count = NumNeighbors;
            if constexpr (IdFormat == ENeighborIdFormat::Ui32) {
                const ui32* const row = Ui32Row(id);
                for (ui32 i = 0; i < count; ++i) {
                    func(row[i]);
                }
            } else {
                TNeighborRowUnpacker<IdFormat>::ForEach(Payload, RowBitOffset(id), count, BitsPerId, std::forward<TFunc>(func));
            }
        }

        /**
         * @brief Row @p id as a view, reading it as @p IdFormat.
         *
         * For the ui32 format the view points straight into the blob and @p scratch is left alone;
         * for the packed formats the row is unpacked into @p scratch. The view therefore stays
         * valid only until the next call made with the same @p scratch: to hold two rows at once,
         * pass a separate scratch buffer for each.
         */
        template <ENeighborIdFormat IdFormat>
        Y_FORCE_INLINE TNeighborsView GetRowAs(const ui32 id, TVector<ui32>& scratch) const {
            if constexpr (IdFormat == ENeighborIdFormat::Ui32) {
                return TNeighborsView(Ui32Row(id), NumNeighbors);
            } else {
                scratch.resize_uninitialized(NumNeighbors);
                ui32* __restrict out = scratch.data();
                ForEachInRowAs<IdFormat>(id, [&out](const ui32 neighborId) { *out++ = neighborId; });
                return TNeighborsView(scratch.data(), NumNeighbors);
            }
        }

        /**
         * @brief Row @p id as a view, for a caller that only learns the format at run time.
         *
         * A caller that knows it at compile time should reach GetRowAs directly, so that the row
         * loop holds one format's unpacking and no dispatch.
         */
        TNeighborsView GetRow(const ui32 id, TVector<ui32>& scratch) const {
            if (Format == ENeighborIdFormat::Ui32) {
                return GetRowAs<ENeighborIdFormat::Ui32>(id, scratch);
            }
            return GetPackedRow(id, scratch);
        }

    private:
        TNeighborsView GetPackedRow(const ui32 id, TVector<ui32>& scratch) const {
            return DispatchByFormat(Format, [&]<ENeighborIdFormat F>() { return GetRowAs<F>(id, scratch); });
        }

        /**
         * @brief Row @p id of a ui32 level, off the base resolved once at construction.
         */
        const ui32* Ui32Row(const ui32 id) const {
            return Ui32Base + size_t(id) * NumNeighbors;
        }

        ui64 RowBitOffset(const ui32 id) const {
            return LevelBitOffset + ui64(id) * RowBitStride;
        }

    private:
        ENeighborIdFormat Format = ENeighborIdFormat::Ui32;
        const ui8* Payload = nullptr;
        const ui32* Ui32Base = nullptr;
        ui64 LevelBitOffset = 0;
        ui64 RowBitStride = 0;
        ui32 NumNeighbors = 0;
        ui32 BitsPerId = 32;
    };

    template <ENeighborIdFormat IdFormat>
    class TBuiltinNeighborRowAccessor {
    public:
        explicit TBuiltinNeighborRowAccessor(TLevelRows rows)
            : Rows(std::move(rows))
        {
            Y_ENSURE(Rows.GetFormat() == IdFormat, "neighbor id format mismatch");
            Scratch.reserve(Rows.GetNumNeighbors());
        }

        ui32 GetNumNeighbors() const {
            return Rows.GetNumNeighbors();
        }

        template <CNeighborVisitor TFunc>
        Y_FORCE_INLINE void ForEachInRow(ui32 id, TFunc&& func) {
            Rows.ForEachInRowAs<IdFormat>(id, std::forward<TFunc>(func));
        }

        TNeighborsView GetRow(ui32 id) {
            return Rows.GetRowAs<IdFormat>(id, Scratch);
        }

    private:
        TLevelRows Rows;
        TVector<ui32> Scratch;
    };

    static_assert(CNeighborRowAccessor<TBuiltinNeighborRowAccessor<ENeighborIdFormat::Ui32>>);
    static_assert(CNeighborRowAccessor<TBuiltinNeighborRowAccessor<ENeighborIdFormat::Ui24>>);
    static_assert(CNeighborRowAccessor<TBuiltinNeighborRowAccessor<ENeighborIdFormat::BitPacked>>);

    /**
     * @brief Collects the unvisited neighbors of a row, reading it as @p IdFormat.
     *
     * The format is a template parameter rather than a member test so that GetLayerNeighbors, which
     * runs once per popped candidate, holds the unpacking loop of one format and nothing else. The
     * dispatch happens once per search in VisitBuiltinNeighborRowAccessor. Ui32 is the default so that the older spelling
     * TNeighborsGetterBase<TSearchContext> keeps naming the format the raw ui32 rows are in.
     */
    template <class TSearchContext, ENeighborIdFormat IdFormat = ENeighborIdFormat::Ui32>
    class TNeighborsGetterBase
        : public TNeighborsGetter<TBuiltinNeighborRowAccessor<IdFormat>, TSearchContext>
    {
        using TBase = TNeighborsGetter<TBuiltinNeighborRowAccessor<IdFormat>, TSearchContext>;

    public:
        TNeighborsGetterBase(const TLevelRows& rows, TSearchContext& context)
            : TBase(TBuiltinNeighborRowAccessor<IdFormat>(rows), context)
        {
        }

        TNeighborsGetterBase(const ui32* level, ui32 numNeighbors, TSearchContext& context)
            requires(IdFormat == ENeighborIdFormat::Ui32)
            : TNeighborsGetterBase(TLevelRows(level, numNeighbors), context)
        {
        }
    };

    template <class TSearchContext, class TFilter, ENeighborIdFormat IdFormat = ENeighborIdFormat::Ui32>
    class TAcornNeighborsGetter
        : public TAcornNeighborsGetterImpl<TBuiltinNeighborRowAccessor<IdFormat>, TSearchContext, TFilter>
    {
        using TBase = TAcornNeighborsGetterImpl<TBuiltinNeighborRowAccessor<IdFormat>, TSearchContext, TFilter>;

    public:
        TAcornNeighborsGetter(const TLevelRows& rows, TSearchContext& context, TFilter& filter)
            : TBase(TBuiltinNeighborRowAccessor<IdFormat>(rows), context, filter)
        {
        }
    };

    template <class TCallback>
    void VisitBuiltinNeighborRowAccessor(const TLevelRows& rows, TCallback&& callback) {
        if (rows.GetFormat() == ENeighborIdFormat::Ui32) {
            std::forward<TCallback>(callback)(TBuiltinNeighborRowAccessor<ENeighborIdFormat::Ui32>(rows));
        } else if (rows.GetFormat() == ENeighborIdFormat::Ui24) {
            std::forward<TCallback>(callback)(TBuiltinNeighborRowAccessor<ENeighborIdFormat::Ui24>(rows));
        } else {
            Y_ENSURE(rows.GetFormat() == ENeighborIdFormat::BitPacked, "unsupported neighbor id format");
            std::forward<TCallback>(callback)(TBuiltinNeighborRowAccessor<ENeighborIdFormat::BitPacked>(rows));
        }
    }

    template <class TSearchContext, class TCallback>
    void VisitBuiltinNeighborsGetter(const TLevelRows& rows, TSearchContext& context, TCallback&& callback) {
        VisitBuiltinNeighborRowAccessor(rows, [&](auto rowAccessor) {
            using TRowAccessor = std::remove_cvref_t<decltype(rowAccessor)>;
            TNeighborsGetter<TRowAccessor, TSearchContext> getter(std::move(rowAccessor), context);
            std::forward<TCallback>(callback)(getter);
        });
    }

    template <class TSearchContext, class TFilter, class TCallback>
    void VisitBuiltinAcornNeighborsGetter(
        const TLevelRows& rows,
        TSearchContext& context,
        TFilter& filter,
        TCallback&& callback
    ) {
        VisitBuiltinNeighborRowAccessor(rows, [&](auto rowAccessor) {
            using TRowAccessor = std::remove_cvref_t<decltype(rowAccessor)>;
            TAcornNeighborsGetterImpl<TRowAccessor, TSearchContext, TFilter> getter(
                std::move(rowAccessor), context, filter
            );
            std::forward<TCallback>(callback)(getter);
        });
    }
} // namespace NHnsw
