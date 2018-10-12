#pragma once

#include <catboost/libs/data_new/target.h>


namespace NCB {
    namespace NDataNewUT {

    template <class T>
    inline TSharedVector<T> ShareVector(TVector<T>&& data) {
        return MakeAtomicShared<TVector<T>>(std::move(data));
    }

    inline TSharedWeights<float> Share(TWeights<float>&& data) {
        return MakeIntrusive<TWeights<float>>(std::move(data));
    }


    void CompareTargets(const TBinClassTarget& lhs, const TBinClassTarget& rhs);

    void CompareTargets(const TMultiClassTarget& lhs, const TMultiClassTarget& rhs);

    void CompareTargets(const TRegressionTarget& lhs, const TRegressionTarget& rhs);

    void CompareTargets(const TGroupwiseRankingTarget& lhs, const TGroupwiseRankingTarget& rhs);

    void CompareTargets(const TGroupPairwiseRankingTarget& lhs, const TGroupPairwiseRankingTarget& rhs);


    void CompareTargetDataProviders(const TTargetDataProviders& lhs, const TTargetDataProviders& rhs);

    }

}

