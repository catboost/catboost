#include "cat_feature.h"
#include <util/digest/city.h>

int CalcCatFeatureHash(const TStringBuf& feature) {
    return CityHash64(feature) & 0xffffffff;
}
