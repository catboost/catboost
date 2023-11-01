#include "cat_feature_perfect_hash.h"

#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/output.h>


template <>
void Out<NCB::TCatFeatureUniqueValuesCounts>(
    IOutputStream& out,
    TTypeTraits<NCB::TCatFeatureUniqueValuesCounts>::TFuncParam counts
) {
    out << counts.OnLearnOnly << ',' << counts.OnAll;
}


template <>
void Out<NCB::TValueWithCount>(IOutputStream &out, TTypeTraits<NCB::TValueWithCount>::TFuncParam valueWithCount) {
    out << "Value="<< valueWithCount.Value << ",Count=" << valueWithCount.Count;
}


namespace NCB {
    bool TCatFeaturePerfectHashDefaultValue::operator==(const TCatFeaturePerfectHashDefaultValue& rhs) const {
        return (SrcValue == rhs.SrcValue) &&
            (DstValueWithCount == rhs.DstValueWithCount) &&
            FuzzyEquals(Fraction, rhs.Fraction);
    }

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

    void TCatFeaturesPerfectHash::UpdateFeaturePerfectHash(
        const TCatFeatureIdx catFeatureIdx,
        TCatFeaturePerfectHash&& perfectHash
    ) {
        CheckHasFeature(catFeatureIdx);

        auto& counts = CatFeatureUniqValuesCountsVector[*catFeatureIdx];

        if (counts.OnAll) {
            // already have some data
            // we must update with data that has not less elements than current
            CB_ENSURE(
                (size_t)counts.OnAll <= perfectHash.GetSize(),
                "Cat feature " << *catFeatureIdx << " has too many unique values ");
        } else {
            // first initialization
            counts.OnLearnOnly = (ui32)perfectHash.GetSize();
        }

        // cast is safe because map from ui32 keys can't have more than Max<ui32>() keys
        counts.OnAll = (ui32)perfectHash.GetSize();

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
        binSaver.AddMulti(CatFeatureUniqValuesCountsVector, FeaturesPerfectHash);
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
