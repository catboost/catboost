#include "dataset.h"

#include <util/generic/xrange.h>

TDataset BuildDataset(const TPool& pool) {
    TDataset data;
    data.Target = pool.Docs.Target;
    data.Weights = pool.Docs.Weight;
    data.QueryId = pool.Docs.QueryId;
    data.SubgroupId = pool.Docs.SubgroupId;
    data.Baseline = pool.Docs.Baseline;
    data.Pairs = pool.Pairs;
    data.HasGroupWeight = pool.MetaInfo.HasGroupWeight;
    return data;
}

void QuantizeTrainPools(
    const TClearablePoolPtrs& pools,
    const TVector<TFloatFeature>& floatFeatures,
    const TVector<int>& ignoredFeatures,
    size_t oneHotMaxSize,
    NPar::TLocalExecutor& localExecutor,
    TDataset* learnData,
    TVector<TDataset>* testDatasets
) {
    THashSet<int> catFeatures(pools.Learn->CatFeatures.begin(), pools.Learn->CatFeatures.end());

    PrepareAllFeaturesLearn(
        catFeatures,
        floatFeatures,
        ignoredFeatures,
        /*ignoreRedundantCatFeatures=*/true,
        oneHotMaxSize,
        /*clearPoolAfterBinarization=*/pools.AllowClearLearn,
        localExecutor,
        /*select=*/{},
        &pools.Learn->Docs,
        &(learnData->AllFeatures)
    );

    testDatasets->resize(pools.Test.size());

    for (auto testDataIdx : xrange(pools.Test.size())) {
        PrepareAllFeaturesTest(
            catFeatures,
            floatFeatures,
            learnData->AllFeatures,
            /*allowNansOnlyInTest=*/false,
            /*clearPoolAfterBinarization=*/pools.AllowClearTest,
            localExecutor,
            /*select=*/{},
            &(pools.Test[testDataIdx]->Docs),
            &((*testDatasets)[testDataIdx].AllFeatures)
        );
    }
}
