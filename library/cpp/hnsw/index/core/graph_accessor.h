#pragma once

#include "neighbors_getter.h"

#include <util/memory/blob.h>
#include <util/system/types.h>

#include <type_traits>
#include <utility>
#include <stddef.h>

namespace NHnsw {
    template <class TGraph>
    using TGraphLevelReader = std::remove_cvref_t<decltype(
        std::declval<const TGraph&>().CreateLevelReader(ui32{})
    )>;

    /**
     * A level reader returns either graph-owned memory or @p scratch. The returned view must not
     * refer to memory owned by the reader itself, because readers are short-lived temporary objects.
     */
    template <class TLevelReader>
    concept CGraphLevelReader = requires(TLevelReader& reader, ui32 id, TVector<ui32>& scratch) {
        requires std::is_same_v<decltype(reader.GetRow(id, scratch)), TNeighborsView>;
    };

    template <class TGraph>
    concept CHnswGraph = requires(const TGraph& graph, ui32 level) {
        requires std::is_same_v<decltype(graph.GetNumLevels()), ui32>;
        requires std::is_same_v<decltype(graph.GetNumNeighbors(level)), ui32>;
        graph.CreateLevelReader(level);
    } && CGraphLevelReader<TGraphLevelReader<TGraph>>;

    template <class TCodec>
    using TCodecGraph = std::remove_cvref_t<decltype(
        std::declval<const TCodec&>().Load(std::declval<TBlob>())
    )>;

    template <class TCodec>
    concept CGraphCodec = requires(const TCodec& codec, TBlob blob) {
        requires std::is_same_v<decltype(codec.Load(std::move(blob))), TCodecGraph<TCodec>>;
        requires CHnswGraph<TCodecGraph<TCodec>>;
    };

    /**
     * @brief Adapts a concrete graph codec to the graph-access interface used by HNSW search.
     */
    template <CGraphCodec TCodec>
    class TCodecGraphAccessor {
        using TGraph = TCodecGraph<TCodec>;

    public:
        TCodecGraphAccessor(TBlob blob, const TCodec& codec)
            : Graph_(codec.Load(std::move(blob)))
        {
        }

        ui32 GetNumLevels() const {
            return Graph_.GetNumLevels();
        }

        ui32 GetNumNeighbors(ui32 level) const {
            return Graph_.GetNumNeighbors(level);
        }

        size_t GetRowScratchSize() const {
            if constexpr (requires { Graph_.GetRowScratchSize(); }) {
                return Graph_.GetRowScratchSize();
            }
            return GetNumLevels() == 0 ? 0 : GetNumNeighbors(0);
        }

        auto CreateLevelReader(ui32 level) const {
            return Graph_.CreateLevelReader(level);
        }

        template <class TFilter, class TSearchContext, class TCallback>
        void VisitNeighborsGetter(TFilter& filter, TSearchContext& context, TCallback&& callback) const {
            if constexpr (requires { Graph_.VisitNeighborsGetter(filter, context, std::forward<TCallback>(callback)); }) {
                Graph_.VisitNeighborsGetter(filter, context, std::forward<TCallback>(callback));
            } else {
                auto reader = CreateLevelReader(0);
                const ui32 numNeighbors = GetNumNeighbors(0);
                using TRowAccessor = TReaderNeighborRowAccessor<decltype(reader)>;
                TRowAccessor rows(std::move(reader), numNeighbors);
                if constexpr (requires {
                    filter.VisitNeighborsGetter(
                        std::move(rows), context, std::forward<TCallback>(callback)
                    );
                }) {
                    filter.VisitNeighborsGetter(
                        std::move(rows), context, std::forward<TCallback>(callback)
                    );
                } else {
                    TNeighborsGetter<TRowAccessor, TSearchContext> getter(std::move(rows), context);
                    std::forward<TCallback>(callback)(getter);
                }
            }
        }

    private:
        TGraph Graph_;
    };
} // namespace NHnsw
