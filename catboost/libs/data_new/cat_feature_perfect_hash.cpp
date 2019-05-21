#include "cat_feature_perfect_hash.h"

#include <util/generic/xrange.h>
#include <util/stream/output.h>


template <>
void Out<NCB::TCatFeatureUniqueValuesCounts>(IOutputStream& out, NCB::TCatFeatureUniqueValuesCounts counts) {
    out << counts.OnLearnOnly << ',' << counts.OnAll;
}


template <>
void Out<NCB::TValueWithCount>(IOutputStream& out, NCB::TValueWithCount valueWithCount) {
    out << "Value="<< valueWithCount.Value << ",Count=" << valueWithCount.Count;
}


namespace NCB {

    bool TCatFeaturesPerfectHash::operator==(const TCatFeaturesPerfectHash& rhs) const {
        if (CatFeatureUniqValuesCountsVector != rhs.CatFeatureUniqValuesCountsVector) {
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

    bool TCatFeaturesPerfectHash::IsSupersetOf(const TCatFeaturesPerfectHash& rhs) const {
        if (this == &rhs) { // shortcut
            return true;
        }

        const size_t rhsSize = rhs.CatFeatureUniqValuesCountsVector.size();
        if (rhsSize > CatFeatureUniqValuesCountsVector.size()) {
            return false;
        }

        for (auto catFeatureIdx : xrange(rhsSize)) {
            const auto& counts = CatFeatureUniqValuesCountsVector[catFeatureIdx];
            const auto& rhsCounts = rhs.CatFeatureUniqValuesCountsVector[catFeatureIdx];
            if (rhsCounts.OnLearnOnly != counts.OnLearnOnly) {
                return false;
            }
            if (rhsCounts.OnAll > counts.OnAll) {
                return false;
            }
        }

        if (!HasHashInRam) {
            Load();
        }
        if (!rhs.HasHashInRam) {
            rhs.Load();
        }

        // count differences are ok
        for (auto catFeatureIdx : xrange(rhsSize)) {
            const auto& featurePerfectHash = FeaturesPerfectHash[catFeatureIdx];

            for (const auto& [hashedCatValue, valueWithCount] : rhs.FeaturesPerfectHash[catFeatureIdx]) {
                const auto it = featurePerfectHash.find(hashedCatValue);
                if (it == featurePerfectHash.end()) {
                    return false;
                } else if (valueWithCount.Value != it->second.Value) {
                    return false;
                }
            }
        }

        return true;
    }


    void TCatFeaturesPerfectHash::UpdateFeaturePerfectHash(
        const TCatFeatureIdx catFeatureIdx,
        TCatFeaturePerfectHash&& perfectHash
    ) {
        CheckHasFeature(catFeatureIdx);

        auto& counts = CatFeatureUniqValuesCountsVector[*catFeatureIdx];

        if (counts.OnAll) {
            // already have some data
            // we must update with data that has not less elements than current
            Y_VERIFY((size_t)counts.OnAll <= perfectHash.size());
        } else {
            // first initialization
            counts.OnLearnOnly = (ui32)perfectHash.size();
        }

        // cast is safe because map from ui32 keys can't have more than Max<ui32>() keys
        counts.OnAll = (ui32)perfectHash.size();

        if (!HasHashInRam) {
            Load();
        }
        FeaturesPerfectHash[*catFeatureIdx] = std::move(perfectHash);
    }

    int TCatFeaturesPerfectHash::operator&(IBinSaver& binSaver) {
        if (!binSaver.IsReading()) {
            if (!HasHashInRam) {
                Load();
            }
        }
        binSaver.AddMulti(CatFeatureUniqValuesCountsVector, FeaturesPerfectHash, AllowWriteFiles);
        if (binSaver.IsReading()) {
            HasHashInRam = true;
        }
        return 0;
    }

    ui32 TCatFeaturesPerfectHash::CalcCheckSum() const {
        if (!HasHashInRam) {
            Load();
        }
        ui32 checkSum = 0;
        checkSum = UpdateCheckSum(checkSum, CatFeatureUniqValuesCountsVector);
        checkSum = UpdateCheckSum(checkSum, FeaturesPerfectHash);
        return checkSum;
    }
}
