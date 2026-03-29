#pragma once

#include "filter_base.h"
#include "neighbors_getter.h"

#include <util/generic/ptr.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#include <cstddef>

namespace NHnsw {
    class TDefaultFilter {
    public:
        struct TState {};
        using TResult = TFilterResult<TState>;

        bool IsLimitReached() const {
            return false;
        }

        TResult Check(const ui32 id, TState parentState = {}) const {
            Y_UNUSED(id, parentState);
            return {EFilterVerdict::Accept};
        }
    };

    class TFilterAdapter {
    public:
        struct TState {};
        using TResult = TFilterResult<TState>;

        TFilterAdapter(const TFilterBase& filter, EFilterMode filterMode, size_t filterCheckLimit)
            : Filter_(filter)
            , FilterMode_(filterMode)
            , FilterCheckLimit_(filterCheckLimit)
        {}

        template <typename TSearchContext>
        THolder<INeighborsGetter> CreateNeighborsGetter(
            const ui32* level, const ui32 numNeighbors, TSearchContext& context)
        {
            if (FilterMode_ == EFilterMode::ACORN) {
                return MakeHolder<TAcornNeighborsGetter<TSearchContext, TFilterAdapter>>(
                    level, numNeighbors, context, *this);
            } else {
                return MakeHolder<TNeighborsGetterBase<TSearchContext>>(level, numNeighbors, context);
            }
        }

        bool IsLimitReached() const {
            if (FilterMode_ == EFilterMode::NO_FILTER) {
                return false;
            }
            return FilterCheckLimit_ == 0;
        }

        TFilterResult<TState> Check(const ui32 id, TState parentState = {}) {
            Y_UNUSED(parentState);
            if (FilterMode_ == EFilterMode::NO_FILTER || DoCheck(id)) {
                return {EFilterVerdict::Accept};
            }
            return {EFilterVerdict::Explore};
        }

    private:
        bool DoCheck(const ui32 id) {
            if (FilterCheckLimit_ == 0) {
                return false;
            }
            --FilterCheckLimit_;
            return Filter_.Check(id);
        }

    private:
        const TFilterBase& Filter_;
        const EFilterMode FilterMode_;
        size_t FilterCheckLimit_;
    };
} // namespace NHnsw
