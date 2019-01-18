#pragma once

#include <catboost/libs/cat_feature/cat_feature.h>
#include <util/string/builder.h>

template <class T, class U>
inline T CeilDivide(T x, U y) {
    return (x + y - 1) / y;
}

inline int StringToIntHash(const TStringBuf& buf) {
    return CalcCatFeatureHash(buf);
}
