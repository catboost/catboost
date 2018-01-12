#include "cv_data_partition.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/helpers/permutation.h>

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
    CB_ENSURE(foldIdx >= 0 && foldIdx < foldCount);
    TFastRng64 rand(seed);
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
    const size_t docCount = allDocs.GetDocCount();
    // TODO(annaveronika): fix learn and test count when query id is fixed.
    const size_t testCount = (docCount - 1 - foldIdx) / foldCount + 1; // number of foldIdx + n*foldCount in [0, docCount)
    const size_t learnCount = docCount - testCount;
    bool hasQueryId = !learnPool->Docs.QueryId.empty();
    learnPool->Docs.Resize(learnCount, allDocs.GetFactorsCount(), allDocs.GetBaselineDimension(), hasQueryId);
    testPool->Docs.Resize(testCount, allDocs.GetFactorsCount(), allDocs.GetBaselineDimension(), hasQueryId);

    size_t learnIdx = 0;
    size_t testIdx = 0;
    for (size_t i = 0; i < docCount; ++i) {
        if (i % foldCount == foldIdx) {
            testPool->Docs.AssignDoc(testIdx, allDocs, i);
            ++testIdx;
        } else {
            learnPool->Docs.AssignDoc(learnIdx, allDocs, i);
            ++learnIdx;
        }
    }

    if (reverseCv) {
        learnPool->Docs.Swap(testPool->Docs);
    }
    MATRIXNET_INFO_LOG << "Learn docs: " << learnPool->Docs.GetDocCount()
                       << ", test docs: " << testPool->Docs.GetDocCount() << Endl;
}
