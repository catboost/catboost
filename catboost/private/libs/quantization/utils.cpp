#include "utils.h"

#include "grid_creator.h"

#include <catboost/libs/helpers/cpu_random.h>

#include <util/generic/utility.h>


template <class T>
static TVector<T> SampleVector(const TVector<T>& vec,
                               ui32 size,
                               ui64 seed) {
    TRandom random(seed);
    TVector<T> result(size);
    for (ui32 i = 0; i < size; ++i) {
        result[i] = vec[(random.NextUniformL() % vec.size())];
    }
    return result;
};


TVector<float> NCB::BuildBorders(const TVector<float>& floatFeature, const ui32 seed,
                                 const NCatboostOptions::TBinarizationOptions& config){
    TOnCpuGridBuilderFactory gridBuilderFactory;
    ui32 sampleSize = GetSampleSizeForBorderSelectionType(floatFeature.size(),
                                                          config.BorderSelectionType);
    if (sampleSize < floatFeature.size()) {
        auto sampledValues = SampleVector(floatFeature, sampleSize, TRandom::GenerateSeed(seed));
        return TBordersBuilder(gridBuilderFactory, sampledValues)(config);
    } else {
        return TBordersBuilder(gridBuilderFactory, floatFeature)(config);
    }
};
