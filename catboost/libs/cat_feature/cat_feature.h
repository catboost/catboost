#pragma once

#include <util/generic/fwd.h>

int CalcCatFeatureHash(TStringBuf feature) noexcept;

inline float ConvertCatFeatureHashToFloat(int hashVal) {
    return *reinterpret_cast<const float*>(&hashVal);
}

inline int ConvertFloatCatFeatureToIntHash(float feature)  {
    return *reinterpret_cast<const int*>(&feature);
}
