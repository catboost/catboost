#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/text_features/text_processing_collection.h>

TFullModel TrainFloatCatboostModel(int iterations = 5, int seed = 123);

NCB::TDataProviderPtr GetAdultPool();
NCB::TDataProviderPtr GetMultiClassPool();

NCB::TDataProviderPtr GetMultiClassPool();

TFullModel SimpleFloatModel(size_t treeCount = 1);

TFullModel SimpleTextModel(
    TIntrusivePtr<NCB::TTextProcessingCollection> textCollection,
    TConstArrayRef<TVector<TStringBuf>> textFeatures,
    TArrayRef<double> expectedResults
);

TFullModel SimpleDeepTreeModel(size_t treeDepth = 10);

TFullModel SimpleAsymmetricModel();

TFullModel MultiValueFloatModel();

// Deterministically train model that has only 3 categorical features.
TFullModel TrainCatOnlyModel();

TFullModel TrainCatOnlyNoOneHotModel();