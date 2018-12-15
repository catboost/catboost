#include "cat_feature.h"

#include <util/digest/city.h>
#include <util/generic/strbuf.h>

ui32 CalcCatFeatureHash(const TStringBuf feature) noexcept {
    return CityHash64(feature) & 0xffffffff;
}
