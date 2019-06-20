#pragma once

#include <algorithm>
#include <util/generic/set.h>

namespace NCB {
    template <class TSortedStorage>
    inline bool IsSubset(const TSortedStorage& subset, const TSortedStorage& set) {
        return std::includes(set.begin(), set.end(), subset.begin(), subset.end());
    }

    template <class TSortedStorage>
    inline TVector<typename TSortedStorage::value_type> RemoveExisting(
        const TSortedStorage& from,
        const TSortedStorage& what) {

        using T = typename TSortedStorage::value_type;
        TVector<T> result;
        std::set_difference(
            from.begin(),
            from.end(),
            what.begin(),
            what.end(),
            std::inserter(result, result.end()));
        return result;
    }
}
