#include "int_cast.h"

#include <util/generic/cast.h>
#include <util/generic/xrange.h>


TVector<ui32> ToUnsigned(const TVector<int>& src) {
    TVector<ui32> result;
    result.yresize(src.size());
    for (auto idx : xrange(src.size())) {
        result[idx] = SafeIntegerCast<ui32>(src[idx]);
    }
    return result;
}

THashSet<int> ToSigned(const THashSet<ui32>& src) {
    THashSet<int> result;
    for (auto element : src) {
        result.insert(SafeIntegerCast<int>(element));
    }
    return result;
}
