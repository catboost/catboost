#pragma once

#include <cmath>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <util/system/types.h>
#include <util/string/builder.h>
#include <util/generic/ymath.h>

inline ui32 IntLog2(ui32 values) {
    return (ui32)ceil(log2(values));
}

inline bool IsPowerOfTwo(ui32 value) {
    return (1 << IntLog2(value)) == value;
}

template <class T, class U>
inline T CeilDivide(T x, U y) {
    return (x + y - 1) / y;
}

inline int StringToIntHash(const TStringBuf& buf) {
    return CalcCatFeatureHash(buf);
}
