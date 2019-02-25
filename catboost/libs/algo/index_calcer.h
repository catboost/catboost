#pragma once

#include "fold.h"
#include "features_data_helpers.h"
#include "split.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


void SetPermutedIndices(
    const TSplit& split,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    int curDepth,
    const TFold& fold,
    TVector<TIndexType>* indices,
    NPar::TLocalExecutor* localExecutor);

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices);

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty);

TVector<TIndexType> BuildIndices(
    const TFold& fold, // can be empty
    const TSplitTree& tree,
    NCB::TTrainingForCPUDataProviderPtr learnData, // can be nullptr
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData, // can be empty
    NPar::TLocalExecutor* localExecutor);

struct TFullModel;

TVector<ui8> GetModelCompatibleQuantizedFeatures(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    size_t start,
    size_t end);

TVector<ui8> GetModelCompatibleQuantizedFeatures(const TFullModel& model, const NCB::TObjectsDataProvider& objectsData);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const TVector<ui8>& binarizedFeatures,
    size_t treeId);

template <class TGetFeatureDataBeginPtr, class TNumType>
static inline void GetRepackedFeatures(
    int blockFirstIdx,
    int blockLastIdx,
    size_t flatFeatureVectorExpectedSize,
    const THashMap<ui32, ui32>& columnReorderMap,
    const TGetFeatureDataBeginPtr& getFeatureDataBeginPtr,
    const NCB::TFeaturesLayout& featuresLayout,
    TVector<TConstArrayRef<TNumType>>* repackedFeatures,
    TVector<TMaybe<NCB::TPackedBinaryIndex>>* packedIndexes = nullptr)
{

    repackedFeatures->resize(flatFeatureVectorExpectedSize);
    if (packedIndexes != nullptr) {
        packedIndexes->resize(flatFeatureVectorExpectedSize);
    }
    const int blockSize = blockLastIdx - blockFirstIdx;
    if (columnReorderMap.empty()) {
        for (size_t i = 0; i < flatFeatureVectorExpectedSize; ++i) {
            if (featuresLayout.GetExternalFeaturesMetaInfo()[i].IsAvailable) {
                (*repackedFeatures)[i] = MakeArrayRef(getFeatureDataBeginPtr(i, packedIndexes) + blockFirstIdx, blockSize);
            }
        }
    } else {
        for (const auto& [origIdx, sourceIdx] : columnReorderMap) {
            if (featuresLayout.GetExternalFeaturesMetaInfo()[sourceIdx].IsAvailable) {
                (*repackedFeatures)[origIdx] = MakeArrayRef(getFeatureDataBeginPtr(sourceIdx, packedIndexes) + blockFirstIdx,
                                                            blockSize);
            }
        }
    }
}

const ui8* GetFeatureDataBeginPtr(
    const NCB::TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    ui32 featureIdx,
    int consecutiveSubsetBegin,
    TVector<TMaybe<NCB::TPackedBinaryIndex>>* packedIdx);
