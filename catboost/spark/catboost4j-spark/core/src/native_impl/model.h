#pragma once

#include <catboost/libs/cat_feature/cat_feature.h>

#include <catboost/libs/model/enums.h>
#include <catboost/libs/model/model.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


template <class T>
void CalcOnSparkFeatureVector(
    const TFullModel& model,
    TConstArrayRef<T> featureValuesFromSpark,
    TArrayRef<double> result
) {
    TVector<float> floatFeaturesValues;
    floatFeaturesValues.yresize(model.GetNumFloatFeatures());
    for (const auto& floatFeatureMetaData : model.ModelTrees->GetFloatFeatures()) {
        floatFeaturesValues[floatFeatureMetaData.Position.Index]
            = featureValuesFromSpark[floatFeatureMetaData.Position.FlatIndex];
    }

    TVector<int> catFeaturesValues;
    catFeaturesValues.yresize(model.GetNumCatFeatures());
    for (const auto& catFeatureMetaData : model.ModelTrees->GetCatFeatures()) {
        catFeaturesValues[catFeatureMetaData.Position.Index]
            = CalcCatFeatureHashInt(
                ToString(int(featureValuesFromSpark[catFeatureMetaData.Position.FlatIndex]))
            );
    }
    model.Calc(floatFeaturesValues, catFeaturesValues, result);
}
