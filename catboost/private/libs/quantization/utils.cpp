#include "utils.h"

#include "grid_creator.h"

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/sample.h>

#include <util/generic/utility.h>


using namespace NCB;


template <class T>
static TVector<T> SampleArray(TConstArrayRef<T> vec,
                              ui32 size,
                              ui64 seed) {
    TRandom random(seed);
    TVector<T> result(size);
    for (ui32 i = 0; i < size; ++i) {
        result[i] = vec[(random.NextUniformL() % vec.size())];
    }
    return result;
};

TArraySubsetIndexing<ui32> NCB::GetArraySubsetForBuildBorders(ui32 objectCount,
                                                              EBorderSelectionType borderSelectionType,
                                                              bool isRandomShuffled,
                                                              ui32 slowSubsetSize,
                                                              TRestorableFastRng64* rand) {

    const ui32 sampleSize = GetSampleSizeForBorderSelectionType(
        objectCount,
        borderSelectionType,
        slowSubsetSize
    );
    TArraySubsetIndexing<ui32> subsetIndexing;
    if (sampleSize < objectCount) {
        if (isRandomShuffled) {
            // just get first sampleSize elements
            TVector<TSubsetBlock<ui32>> blocks = {TSubsetBlock<ui32>({0, sampleSize}, 0)};
            subsetIndexing = TArraySubsetIndexing<ui32>(
                TRangesSubset<ui32>(sampleSize, std::move(blocks))
            );
        } else {
            TIndexedSubset<ui32> randomShuffle = SampleIndices<ui32>(objectCount, sampleSize, rand);
            subsetIndexing = TArraySubsetIndexing<ui32>(std::move(randomShuffle));
        }
    } else {
        subsetIndexing = TArraySubsetIndexing<ui32>(TFullSubset<ui32>(objectCount));
    }
    return subsetIndexing;
}

TVector<float> NCB::BuildBorders(TConstArrayRef<float> floatFeature, const ui32 seed,
                                 const NCatboostOptions::TBinarizationOptions& config){
    TOnCpuGridBuilderFactory gridBuilderFactory;
    ui32 sampleSize = GetSampleSizeForBorderSelectionType(floatFeature.size(),
                                                          config.BorderSelectionType);
    if (sampleSize < floatFeature.size()) {
        auto sampledValues = SampleArray(floatFeature, sampleSize, TRandom::GenerateSeed(seed));
        return TBordersBuilder(gridBuilderFactory, sampledValues)(config);
    } else {
        return TBordersBuilder(gridBuilderFactory, floatFeature)(config);
    }
};
