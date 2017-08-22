#pragma once
#include <util/generic/strbuf.h>

int CalcCatFeatureHash(const TStringBuf& feature);

inline float ConvertCatFeatureHashToFloat(int hashVal) {
    return *reinterpret_cast<const float*>(&hashVal);
}

inline int ConvertFloatCatFeatureToIntHash(float feature)  {
    return *reinterpret_cast<const int*>(&feature);
}
