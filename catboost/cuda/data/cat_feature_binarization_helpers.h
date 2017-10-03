#pragma once

#include "load_data.h"
#include <util/system/fs.h>

class TCatFeatureBinarizationHelpers {
public:
    using TSimpleCatFeatureBinarizationInfo = TDataProviderBuilder::TSimpleCatFeatureBinarizationInfo;

    static void SaveCatFeatureBinarization(const TSimpleCatFeatureBinarizationInfo& catFeatureBinarizations,
                                           IOutputStream* output) {
        ::Save(output, catFeatureBinarizations);
    }

    static void SaveCatFeatureBinarization(const TSimpleCatFeatureBinarizationInfo& catFeatureBinarizations,
                                           TString fileName) {
        TOFStream out(fileName);
        SaveCatFeatureBinarization(catFeatureBinarizations, &out);
    }

    static TSimpleCatFeatureBinarizationInfo LoadCatFeatureBinarization(IInputStream* input) {
        TSimpleCatFeatureBinarizationInfo catFeatureBinarization;
        ::Load(input, catFeatureBinarization);
        return catFeatureBinarization;
    }

    //store hash = result[i][bin] is catFeatureHash for feature catFeatures[i]
    static yvector<yvector<int>> MakeInverseCatFeatureIndex(const yvector<ui32>& catFeatures,
                                                            const TSimpleCatFeatureBinarizationInfo& info) {
        yvector<yvector<int>> result(catFeatures.size());
        for (ui32 i = 0; i < catFeatures.size(); ++i) {
            if (info.has(catFeatures[i])) {
                const auto& binarization = info.at(catFeatures[i]);
                result[i].resize(binarization.size());
                for (const auto& entry : binarization) {
                    result[i][entry.second] = entry.first;
                }
            }
        }
        return result;
    }

    //store hash = result[i][bin] is catFeatureHash for feature catFeatures[i]
    static yvector<yvector<int>> MakeInverseCatFeatureIndex(const yvector<ui32>& catFeatures,
                                                            TString inputFile) {
        TSimpleCatFeatureBinarizationInfo binarizationInfo;
        if (NFs::Exists(inputFile)) {
            TIFStream inputStream(inputFile);
            binarizationInfo = LoadCatFeatureBinarization(&inputStream);
        }
        return MakeInverseCatFeatureIndex(catFeatures,
                                          binarizationInfo);
    }
};
