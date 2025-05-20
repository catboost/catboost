#pragma once

#include <util/system/types.h>

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

} // namespace NHnsw
