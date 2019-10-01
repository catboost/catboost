#include "util.h"

#include <util/generic/hash_set.h>
#include <util/system/yassert.h>


namespace NCB {

    bool EqualAsMultiSets(TConstArrayRef<TPair> lhs, TConstArrayRef<TPair> rhs) {
        THashMultiSet<TPair> lhsCopy(lhs.begin(), lhs.end());
        THashMultiSet<TPair> rhsCopy(rhs.begin(), rhs.end());
        return lhsCopy == rhsCopy;
    }

}
