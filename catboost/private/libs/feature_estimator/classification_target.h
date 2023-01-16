#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


namespace NCB {
    struct TClassificationTarget : public TThrRefBase {
        TClassificationTarget(TVector<ui32>&& classes, ui32 numClasses)
        : Classes(std::move(classes))
        , NumClasses(numClasses)
        {}

    public:
        TVector<ui32> Classes;
        ui32 NumClasses;
    };

    using TClassificationTargetPtr = TIntrusivePtr<TClassificationTarget>;
}
