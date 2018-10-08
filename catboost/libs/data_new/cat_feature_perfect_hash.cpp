#include "cat_feature_perfect_hash.h"


namespace NCB {

    bool TCatFeaturesPerfectHash::operator==(const TCatFeaturesPerfectHash& rhs) const {
        if (CatFeatureUniqueValues != rhs.CatFeatureUniqueValues) {
            return false;
        }

        if (!HasHashInRam) {
            Load();
        }
        if (!rhs.HasHashInRam) {
            rhs.Load();
        }
        return FeaturesPerfectHash == rhs.FeaturesPerfectHash;
    }

}
