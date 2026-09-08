#pragma once

#include "filter_base.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <library/cpp/hnsw/helpers/neighbor_id_format.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

namespace NHnsw {
    using TNeighborsView = TArrayRef<const ui32>;

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
        template <ENeighborIdFormat IdFormat, class TFunc>
        Y_FORCE_INLINE void ForEachInRowAs(const ui32 id, TFunc&& func) const {
            // A local, so that a func writing to memory cannot force a re-read of the row width.
            const ui32 count = NumNeighbors;
            if constexpr (IdFormat == ENeighborIdFormat::Ui32) {
                const ui32* const row = Ui32Row(id);
                for (ui32 i = 0; i < count; ++i) {
                    func(row[i]);
                }
            } else {
                TNeighborRowUnpacker<IdFormat>::ForEach(Payload, RowBitOffset(id), count, BitsPerId, func);
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

    class INeighborsGetter {
    public:
        virtual ~INeighborsGetter() = default;

        virtual bool IsPrefiltered() const {
            return false;
        }

        virtual TNeighborsView GetLayerNeighbors(const ui32 id) = 0;
    };

    /**
     * @brief Collects the unvisited neighbors of a row, reading it as @p IdFormat.
     *
     * The format is a template parameter rather than a member test so that GetLayerNeighbors, which
     * runs once per popped candidate, holds the unpacking loop of one format and nothing else. The
     * dispatch happens once per search, in MakeNeighborsGetter, and lands in the vtable slot the
     * virtual call already goes through. Ui32 is the default so that the older spelling
     * TNeighborsGetterBase<TSearchContext> keeps naming the format the raw ui32 rows are in.
     */
    template <typename TSearchContext, ENeighborIdFormat IdFormat = ENeighborIdFormat::Ui32>
    class TNeighborsGetterBase: public INeighborsGetter {
    public:
        TNeighborsGetterBase(const TLevelRows& rows, TSearchContext& context)
            : Rows(rows)
            , Context(context)
        {
            Y_ENSURE(
                Rows.GetFormat() == IdFormat,
                "a neighbors getter reading neighbor id format " << IdFormat
                << " cannot walk an index stored in format " << Rows.GetFormat()
            );
            NeighborsBuffer.reserve(Rows.GetNumNeighbors());
        }

        TNeighborsGetterBase(const ui32* level, const ui32 numNeighbors, TSearchContext& context)
            requires(IdFormat == ENeighborIdFormat::Ui32)
            : TNeighborsGetterBase(TLevelRows(level, numNeighbors), context)
        {
        }

        TNeighborsView GetLayerNeighbors(const ui32 id) override {
            return CollectUnvisited(id);
        }

    protected:
        TNeighborsView GetRow(const ui32 id) {
            return Rows.GetRowAs<IdFormat>(id, RowBuffer);
        }

        size_t GetNumNeighbors() const {
            return Rows.GetNumNeighbors();
        }

        TNeighborsView PrefilterVisited(TNeighborsView neighbors) {
            NeighborsBuffer.clear();
            for (ui32 id: neighbors) {
                if (Context.TryMarkVisited(id)) {
                    NeighborsBuffer.push_back(id);
                }
            }
            return NeighborsBuffer;
        }

    private:
        TNeighborsView CollectUnvisited(const ui32 id) {
            NeighborsBuffer.clear();
            Rows.ForEachInRowAs<IdFormat>(id, [this](const ui32 neighborId) {
                if (Context.TryMarkVisited(neighborId)) {
                    NeighborsBuffer.push_back(neighborId);
                }
            });
            return NeighborsBuffer;
        }

    private:
        TLevelRows Rows;
        TSearchContext& Context;
        TVector<ui32> RowBuffer;
        TVector<ui32> NeighborsBuffer;
    };

    /**
     * @brief Builds the TNeighborsGetterBase instantiation that reads @p rows.
     *
     * Resolving the neighbor id format here, once per search, is what keeps it off the row path.
     */
    template <typename TSearchContext>
    THolder<INeighborsGetter> MakeNeighborsGetter(const TLevelRows& rows, TSearchContext& context) {
        return DispatchByFormat(rows.GetFormat(), [&]<ENeighborIdFormat F>() -> THolder<INeighborsGetter> {
            return MakeHolder<TNeighborsGetterBase<TSearchContext, F>>(rows, context);
        });
    }

    template <typename TSearchContext, typename TFilter, ENeighborIdFormat IdFormat = ENeighborIdFormat::Ui32>
    class TAcornNeighborsGetter: public TNeighborsGetterBase<TSearchContext, IdFormat> {
    public:
        TAcornNeighborsGetter(const TLevelRows& rows, TSearchContext& context, TFilter& filter)
            : TNeighborsGetterBase<TSearchContext, IdFormat>(rows, context)
            , Filter(filter)
        {
            const size_t numNeighbors = this->GetNumNeighbors();
            AcornNeighbors.resize(numNeighbors * numNeighbors, 0);
            SecondHopStorage.resize(numNeighbors, 0);
        }

        bool IsPrefiltered() const override {
            return true;
        }

        TNeighborsView GetLayerNeighbors(const ui32 id) override {
            ui32 acornCount = 0;
            ScanNeighbors(id, acornCount, /*isFirstHop*/ true);

            const size_t numSecondHops = this->GetNumNeighbors() - acornCount;

            for (size_t i = 0; i < numSecondHops; ++i) {
                ScanNeighbors(SecondHopStorage[i], acornCount, /*isFirstHop*/ false);
            }

            return this->PrefilterVisited(TNeighborsView{AcornNeighbors.data(), acornCount});
        }

    private:
        void ScanNeighbors(const ui32 id, ui32& acornCount, bool isFirstHop) {
            const TNeighborsView neighbors = this->GetRow(id);
            for (size_t i = 0; i < neighbors.size() && !Filter.IsLimitReached(); ++i) {
                ui32 neighbor = neighbors[i];

                if (isFirstHop && !SeenInFirstHop.Insert(neighbor)) {
                    continue;
                }

                if (!isFirstHop && SeenInFirstHop.Has(neighbor)) {
                    continue;
                }

                bool passesFilter = true;
                if (const auto* filterOk = FilterResult.FindPtr(neighbor)) {
                    passesFilter = *filterOk;
                    if (passesFilter || !isFirstHop) {
                        continue;
                    }
                } else {
                    passesFilter = (Filter.Check(neighbor).Verdict == EFilterVerdict::Accept);
                    FilterResult[neighbor] = passesFilter;
                }

                if (passesFilter) {
                    AcornNeighbors[acornCount++] = neighbor;
                } else if (isFirstHop) {
                    SecondHopStorage[i - acornCount] = neighbor;
                }
            }
        }

    private:
        TFilter& Filter;
        TVector<ui32> AcornNeighbors;

        TDenseHash<ui32, bool> FilterResult;
        TDenseHashSet<ui32> SeenInFirstHop;

        TVector<ui32> SecondHopStorage;
    };

    /**
     * @brief Builds the TAcornNeighborsGetter instantiation that reads @p rows.
     */
    template <typename TSearchContext, typename TFilter>
    THolder<INeighborsGetter> MakeAcornNeighborsGetter(
        const TLevelRows& rows,
        TSearchContext& context,
        TFilter& filter
    ) {
        return DispatchByFormat(rows.GetFormat(), [&]<ENeighborIdFormat F>() -> THolder<INeighborsGetter> {
            return MakeHolder<TAcornNeighborsGetter<TSearchContext, TFilter, F>>(rows, context, filter);
        });
    }

} // namespace NHnsw
