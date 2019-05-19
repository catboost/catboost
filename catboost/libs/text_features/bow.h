#pragma once

#include "text_dataset.h"
#include <catboost/libs/feature_estimator/feature_estimator.h>

namespace NCB {


    class TBagOfWordsEstimator final : public IFeatureEstimator {
    public:
        TBagOfWordsEstimator(
            TTextDataSetPtr learnTexts,
            TVector<TTextDataSetPtr> testTexts)
            :  LearnTexts({learnTexts})
            , TestTexts(std::move(testTexts))
            , Dictionary(learnTexts->GetDictionary()){

        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            const ui32 featureCount = Dictionary.Size();
            TEstimatedFeaturesMeta meta;
            meta.Type = TVector<EFeatureCalculatorType>(featureCount, EFeatureCalculatorType::BoW);
            meta.FeaturesCount = featureCount;
            meta.UniqueValuesUpperBoundHint = TVector<ui32>(featureCount, 2);
            return meta;
        }

        void ComputeFeatures(TCalculatedFeatureVisitor learnVisitor,
                             TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
                             NPar::TLocalExecutor* executor) const override;


    protected:

        void Calc(NPar::TLocalExecutor& executor,
                  TConstArrayRef<TTextDataSetPtr> dataSets,
                  TConstArrayRef<TCalculatedFeatureVisitor> visitors) const;

    private:
        TVector<TTextDataSetPtr> LearnTexts;
        TVector<TTextDataSetPtr> TestTexts;
        const IDictionary& Dictionary;
    };

}
