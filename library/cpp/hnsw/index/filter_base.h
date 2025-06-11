#pragma once

#include <util/system/types.h>

#include <stddef.h>

namespace NHnsw {

    enum class EFilterMode: ui32 {
        NO_FILTER = 0,
        FILTER_NEAREST = 1,
        ACORN = 2,
    };

    class TFilterBase {
    public:
        virtual ~TFilterBase() = default;

        virtual bool Check(const ui32 /*id*/) const {
            return true;
        }
    };

    class TFilterWithLimit {
    public:
        TFilterWithLimit(const TFilterBase& filter, const size_t filterCheckLimit)
            : Filter(filter)
            , FilterCheckLimit(filterCheckLimit)
        {
        }

        bool Check(const ui32 id) const {
            if (FilterCheckLimit == 0) {
                return false;
            }
            --FilterCheckLimit;
            return Filter.Check(id);
        }

        bool IsLimitReached() const {
            return FilterCheckLimit == 0;
        }

    private:
        const TFilterBase& Filter;
        mutable size_t FilterCheckLimit;
    };

} // namespace NHnsw
