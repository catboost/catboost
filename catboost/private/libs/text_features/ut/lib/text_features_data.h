#pragma once

#include <catboost/private/libs/text_features/bow.h>
#include <catboost/private/libs/text_features/bm25.h>
#include <catboost/private/libs/text_features/feature_calcer.h>
#include <catboost/private/libs/text_features/naive_bayesian.h>
#include <catboost/private/libs/text_features/text_processing_collection.h>
#include <catboost/private/libs/text_processing/tokenizer.h>
#include <catboost/private/libs/text_processing/dictionary.h>
#include <catboost/private/libs/text_processing/ut/lib/text_processing_data.h>

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>

namespace NCBTest {
    TIntrusivePtr<NCB::TMultinomialNaiveBayes> CreateBayes(
        const TTokenizedTextFeature& features,
        TConstArrayRef<ui32> target,
        ui32 numClasses
    );

    TIntrusivePtr<NCB::TBM25> CreateBm25(
        const TTokenizedTextFeature& features,
        TConstArrayRef<ui32> target,
        ui32 numClasses
    );

    TIntrusivePtr<NCB::TBagOfWordsCalcer> CreateBoW(const NCB::TDictionaryPtr& dictionaryPtr);

    void CreateTextDataForTest(
        TVector<TTextFeature>* features,
        TMap<ui32, TTokenizedTextFeature>* tokenizedFeatures,
        TVector<NCB::TDigitizer>* digitizers,
        TVector<NCB::TTextFeatureCalcerPtr>* calcers,
        TVector<TVector<ui32>>* perFeatureDigitizers,
        TVector<TVector<ui32>>* perTokenizedFeatureCalcers
    );

    TIntrusivePtr<NCB::TTextProcessingCollection> CreateTextProcessingCollectionForTest();
}
