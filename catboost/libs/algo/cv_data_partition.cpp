#include "cv_data_partition.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/random/fast.h>
#include <util/random/shuffle.h>

void BuildCvPools(
    int foldIdx,
    int foldCount,
    bool reverseCv,
    int seed,
    TPool* learnPool,
    TPool* testPool)
{
    CB_ENSURE(foldIdx >= 0 && foldIdx < foldCount);
    TFastRng64 rand(seed);
    Shuffle(learnPool->Docs.begin(), learnPool->Docs.end(), rand);
    testPool->CatFeatures = learnPool->CatFeatures;

    foldIdx = foldIdx % foldCount;
    TPool allDocs;
    allDocs.Docs.swap(learnPool->Docs);

    for (int i = 0; i < allDocs.Docs.ysize(); ++i) {
        if (i % foldCount == foldIdx) {
            testPool->Docs.emplace_back();
            testPool->Docs.back().Swap(allDocs.Docs[i]);
        } else {
            learnPool->Docs.emplace_back();
            learnPool->Docs.back().Swap(allDocs.Docs[i]);
        }
    }

    if (reverseCv) {
        swap(learnPool->Docs, testPool->Docs);
    }
    MATRIXNET_INFO_LOG << "Learn docs: " << learnPool->Docs.ysize()
                       << ", test docs: " << testPool->Docs.ysize() << Endl;
}
