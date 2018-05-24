#include "cat_feature.h"

#include <util/digest/city.h>
#include <util/generic/strbuf.h>

int CalcCatFeatureHash(const TStringBuf feature) noexcept {
    return CityHash64(feature) & 0xffffffff;
}
