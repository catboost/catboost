#pragma once

#include <util/generic/hash_set.h>
#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>


namespace NCB {

    template <class T>
    bool EqualAsMultiSets(TConstArrayRef<T> lhs, TConstArrayRef<T> rhs) {
        THashMultiSet<T> lhsCopy(lhs.begin(), lhs.end());
        THashMultiSet<T> rhsCopy(rhs.begin(), rhs.end());
        return lhsCopy == rhsCopy;
    }

}
