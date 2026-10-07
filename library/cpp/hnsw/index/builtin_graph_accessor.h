#pragma once

#include "builtin_neighbors_getter.h"
#include "index_reader.h"
#include <library/cpp/hnsw/index/core/graph_accessor.h>
#include <library/cpp/hnsw/index/layout/index_layout.h>

#include <util/memory/blob.h>
#include <util/generic/fwd.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

#include <cstdint>
#include <type_traits>
#include <stddef.h>
#include <utility>

namespace NHnsw {
    /** @brief Graph accessor for the built-in ui32 and packed HNSW formats. */
    class TBuiltinGraphAccessor {
    public:
        class TLevelReader {
        public:
            explicit TLevelReader(TLevelRows rows)
                : Rows_(std::move(rows))
            {
            }

            TNeighborsView GetRow(ui32 id, TVector<ui32>& scratch) {
                return Rows_.GetRow(id, scratch);
            }

        private:
            TLevelRows Rows_;
        };

        explicit TBuiltinGraphAccessor(TBlob blob)
            : TBuiltinGraphAccessor(std::move(blob), THnswIndexReader{})
        {
        }

        template <class TIndexReader>
            requires requires(const TIndexReader& reader, const TBlob& blob, THnswIndexLayout* layout) {
                reader.ReadIndex(blob, layout);
            }
        TBuiltinGraphAccessor(TBlob blob, const TIndexReader& indexReader)
            : Data_(std::move(blob))
        {
            indexReader.ReadIndex(Data_, &Layout_);
            const std::uintptr_t blobBegin = reinterpret_cast<std::uintptr_t>(Data_.Begin());
            const std::uintptr_t blobEnd = reinterpret_cast<std::uintptr_t>(Data_.End());
            const std::uintptr_t payload = reinterpret_cast<std::uintptr_t>(Layout_.Payload);
            Y_ENSURE(
                Layout_.Levels.empty() || (payload >= blobBegin && payload <= blobEnd),
                "hnsw index layout payload must point into its source blob"
            );
        }

        ui32 GetNumLevels() const {
            return Layout_.Levels.size();
        }

        ui32 GetNumNeighbors(ui32 level) const {
            return Layout_.Levels[level].NumNeighbors;
        }

        size_t GetRowScratchSize() const {
            return Layout_.Format == ENeighborIdFormat::Ui32 || Layout_.Levels.empty()
                ? 0
                : Layout_.Levels[0].NumNeighbors;
        }

        TLevelReader CreateLevelReader(ui32 level) const {
            return TLevelReader(GetLevelRows(level));
        }

        template <class TFilter, class TSearchContext, class TCallback>
        void VisitNeighborsGetter(TFilter& filter, TSearchContext& context, TCallback&& callback) const {
            const TLevelRows rows = GetLevelRows(0);
            if constexpr (requires { filter.VisitNeighborsGetter(rows, context, std::forward<TCallback>(callback)); }) {
                filter.VisitNeighborsGetter(rows, context, std::forward<TCallback>(callback));
            } else {
                VisitBuiltinNeighborRowAccessor(rows, [&]<CNeighborRowAccessor TRowAccessor>(TRowAccessor rowAccessor) {
                    if constexpr (requires {
                        filter.VisitNeighborsGetter(
                            std::move(rowAccessor), context, std::forward<TCallback>(callback)
                        );
                    }) {
                        filter.VisitNeighborsGetter(
                            std::move(rowAccessor), context, std::forward<TCallback>(callback)
                        );
                    } else {
                        TNeighborsGetter<TRowAccessor, TSearchContext> getter(std::move(rowAccessor), context);
                        std::forward<TCallback>(callback)(getter);
                    }
                });
            }
        }

        const ui32* GetNeighbors(ui32 level, ui32 id) const {
            Y_ENSURE(
                Layout_.Format == ENeighborIdFormat::Ui32,
                "raw neighbor rows are available only for the ui32 neighbor id format"
            );
            const auto& levelLayout = Layout_.Levels[level];
            return reinterpret_cast<const ui32*>(Layout_.Payload + (levelLayout.BitOffset >> 3))
                + size_t(id) * levelLayout.NumNeighbors;
        }

    private:
        TLevelRows GetLevelRows(ui32 level) const {
            return TLevelRows(Layout_.Format, Layout_.Payload, Layout_.BitsPerId, Layout_.Levels[level]);
        }

    private:
        TBlob Data_;
        THnswIndexLayout Layout_;
    };
} // namespace NHnsw
