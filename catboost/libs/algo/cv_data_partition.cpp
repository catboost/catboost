#include "cv_data_partition.h"
#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/data_split.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/logging/logging.h>

#include <util/random/fast.h>

#include <library/threading/local_executor/local_executor.h>

void BuildCvPools(
    int foldIdx,
    int foldCount,
    bool reverseCv,
    int seed,
    int threadCount,
    TPool* learnPool,
    TPool* testPool)
{
    // TODO(annaveronika): one split for all cv.
    CB_ENSURE(foldIdx >= 0 && foldIdx < foldCount);
    CB_ENSURE(learnPool->Docs.GetDocCount() > 1, "Not enough documents for cross validataion");

    TRestorableFastRng64 rand(seed);
    TVector<ui64> permutation;
    permutation.yresize(learnPool->Docs.GetDocCount());
    std::iota(permutation.begin(), permutation.end(), /*starting value*/ 0);
    Shuffle(learnPool->Docs.QueryId, rand, &permutation);
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);
    ApplyPermutation(InvertPermutation(permutation), learnPool, &localExecutor);
    testPool->CatFeatures = learnPool->CatFeatures;

    foldIdx = foldIdx % foldCount;
    TDocumentStorage allDocs;
    allDocs.Swap(learnPool->Docs);
    TVector<TPair> allPairs;
    allPairs.swap(learnPool->Pairs);
    const size_t docCount = allDocs.GetDocCount();

    bool hasQueryId = !allDocs.QueryId.empty();
    TVector<int> queryIndices;
    TVector<TQueryInfo> queryInfo;
    if (hasQueryId) {
        UpdateQueriesInfo(allDocs.QueryId, allDocs.SubgroupId, /*begin=*/0, docCount, &queryInfo);
        queryIndices = GetQueryIndicesForDocs(queryInfo, docCount);
    }

    int testCount = 0;
    TVector<int> foldEndIndices(foldCount + 1, 0);
    for (int i = 1; i < foldEndIndices.ysize(); ++i) {
        foldEndIndices[i] = docCount * i / foldCount;
        if (hasQueryId) {
            foldEndIndices[i] = queryInfo[queryIndices[foldEndIndices[i] - 1]].End;
        }
        int foldSize = foldEndIndices[i] - foldEndIndices[i - 1];
        CB_ENSURE(foldSize > 0, "Not enough documents for cross validataion");
        if (i == foldIdx + 1) {
            testCount = foldSize;
        }
    }
    int learnCount = docCount - testCount;

    bool hasSubgroupId = !allDocs.SubgroupId.empty();
    learnPool->Docs.Resize(learnCount, allDocs.GetFactorsCount(), allDocs.GetBaselineDimension(), hasQueryId, hasSubgroupId);
    testPool->Docs.Resize(testCount, allDocs.GetFactorsCount(), allDocs.GetBaselineDimension(), hasQueryId, hasSubgroupId);

    size_t learnIdx = 0;
    size_t testIdx = 0;

    TVector<ui64> learnPermutation(docCount), testPermutation(docCount);
    for (int i = 1; i < foldEndIndices.ysize(); ++i) {
        if (i == foldIdx + 1) {
            for (int docIdx = foldEndIndices[i - 1]; docIdx < foldEndIndices[i]; ++docIdx) {
                testPool->Docs.AssignDoc(testIdx, allDocs, docIdx);
                testPermutation[testIdx] = docIdx;
                ++testIdx;
            }
        } else {
            for (int docIdx = foldEndIndices[i - 1]; docIdx < foldEndIndices[i]; ++docIdx) {
                learnPool->Docs.AssignDoc(learnIdx, allDocs, docIdx);
                learnPermutation[learnIdx] = docIdx;
                ++learnIdx;
            }
        }
    }
    SplitPairs(allPairs, foldEndIndices[foldIdx], foldEndIndices[foldIdx + 1], &learnPool->Pairs, &testPool->Pairs);
    ApplyPermutationToPairs(InvertPermutation(learnPermutation), &learnPool->Pairs);
    ApplyPermutationToPairs(InvertPermutation(testPermutation), &testPool->Pairs);

    if (reverseCv) {
        learnPool->Docs.Swap(testPool->Docs);
    }
    MATRIXNET_INFO_LOG << "Learn docs: " << learnPool->Docs.GetDocCount()
        << ", test docs: " << testPool->Docs.GetDocCount() << Endl;
}
