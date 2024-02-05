#pragma once

#include <util/generic/strbuf.h>
#include <util/system/types.h>

ui32 CalcCatFeatureHash(const TStringBuf feature) noexcept;

// deprecated, for compatibility, prefer CalcCatFeatureHash in new code
inline int CalcCatFeatureHashInt(const TStringBuf feature) noexcept {
    ui32 hashVal = CalcCatFeatureHash(feature);
    return *reinterpret_cast<int*>(&hashVal);
}

inline float ConvertCatFeatureHashToFloat(ui32 hashVal) noexcept {
    return *reinterpret_cast<const float*>(&hashVal);
}

inline ui32 ConvertFloatCatFeatureToIntHash(float feature) noexcept {
    return *reinterpret_cast<const ui32*>(&feature);
}
