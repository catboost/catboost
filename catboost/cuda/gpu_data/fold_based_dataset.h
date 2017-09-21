#pragma once

#include "binarized_dataset.h"
#include "cat_features_dataset.h"
#include "ctr_helper.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/permutation.h>

inline TCtrTargets<NCudaLib::TSingleMapping> DeviceView(const TCtrTargets<NCudaLib::TMirrorMapping>& mirrorTargets, ui32 devId) {
    TCtrTargets<NCudaLib::TSingleMapping> view;
    view.WeightedTarget = mirrorTargets.WeightedTarget.DeviceView(devId);
    view.BinarizedTarget = mirrorTargets.BinarizedTarget.DeviceView(devId);
    view.Weights = mirrorTargets.Weights.DeviceView(devId);

    view.TotalWeight = mirrorTargets.TotalWeight;
    view.LearnSlice = mirrorTargets.LearnSlice;
    view.TestSlice = mirrorTargets.TestSlice;
    return view;
}

template <NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
class TDataSet: public TGuidHolder {
public:
    using TDocsMapping = NCudaLib::TMirrorMapping;

    static constexpr NCudaLib::EPtrType GetCatFeaturesStoragePtrType() {
        return CatFeaturesStoragePtrType;
    };

    TDataSet(const TDataProvider& dataProvider,
             ui32 permutationId,
             ui32 blockSize)
        : DataProvider(&dataProvider)
        , Permutation(::GetPermutation(dataProvider, permutationId, blockSize))
    {
    }

    //binary features
    const TGpuBinarizedDataSet<TBinaryFeatureGridPolicy>& GetBinaryFeatures() const {
        return *BinaryFeatures;
    }

    //float and one-hot features + simple freq-ctr_type
    const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& GetFeatures() const {
        return *Features;
    }

    //target-ctr_type
    const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& GetTargetCtrs() const {
        return PermutationBasedFeatures;
    }

    //target-ctr_type
    const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const {
        return *CatFeatures;
    }

    bool HasFeature(ui32 featureId) const {
        if (GetBinaryFeatures().HasFeature(featureId)) {
            return true;
        }
        if (GetFeatures().HasFeature(featureId)) {
            return true;
        }
        return GetTargetCtrs().HasFeature(featureId);
    }

    const TDataProvider& GetDataProvider() const {
        CB_ENSURE(DataProvider);
        return *DataProvider;
    }

    const TDataPermutation& GetPermutation() const {
        return Permutation;
    };

    const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const {
        return *CtrTargets;
    }

    bool HasCtrHistoryDataSet() const {
        return LinkedHistoryForCtrs != nullptr;
    }

    const TDataSet& LinkedHistoryForCtr() const {
        CB_ENSURE(HasCtrHistoryDataSet(), "No history dataset found");
        return *LinkedHistoryForCtrs;
    }

    //current permutation order
    const TMirrorBuffer<float>& GetTarget() const {
        return Targets;
    }

    //current permutation order
    const TMirrorBuffer<float>& GetWeights() const {
        return Weights;
    }

    //doc-indexing
    const TMirrorBuffer<ui32>& GetIndices() const {
        return Indices;
    }

    //doc-indexing
    const NCudaLib::TMirrorMapping& GetDocumentsMapping() const {
        return Targets.GetMapping();
    }

    //doc-indexing
    const TMirrorBuffer<ui32>& GetInverseIndices() const {
        return InverseIndices;
    }

private:
    //direct indexing. we will gather them anyway in gradient descent and tree searching,
    // so let's save some memory
    TMirrorBuffer<float> Targets;
    TMirrorBuffer<float> Weights;

    TMirrorBuffer<ui32> Indices;
    TMirrorBuffer<ui32> InverseIndices;

    const TDataProvider* DataProvider;
    TDataPermutation Permutation;

    TSimpleSharedPtr<TGpuBinarizedDataSet<TByteFeatureGridPolicy>> Features;
    TSimpleSharedPtr<TGpuBinarizedDataSet<TBinaryFeatureGridPolicy>> BinaryFeatures;
    TSimpleSharedPtr<TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>> CatFeatures;

    TGpuBinarizedDataSet<TByteFeatureGridPolicy> PermutationBasedFeatures;

    TSimpleSharedPtr<TCtrTargets<NCudaLib::TMirrorMapping>> CtrTargets;
    TSimpleSharedPtr<TDataSet> LinkedHistoryForCtrs;

    template <NCudaLib::EPtrType CatFeatureStoragePtrType>
    friend class TDataSetHoldersBuilder;
};

template <NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
class TDataSetsHolder: public TGuidHolder {
public:
    bool HasFeature(ui32 featureId) const {
        return PermutationDataSets.at(0)->HasFeature(featureId);
    }

    const TDataSet<CatFeaturesStoragePtrType>& GetDataSetForPermutation(ui32 permutationId) const {
        return *PermutationDataSets.at(permutationId);
    }

    const TDataProvider& GetDataProvider() const {
        CB_ENSURE(DataProvider);
        return *DataProvider;
    }

    const TBinarizedFeaturesManager& GetFeaturesManger() const {
        CB_ENSURE(FeaturesManager);
        return *FeaturesManager;
    }

    const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const {
        return PermutationDataSets[0]->GetCatFeatures();
    }

    const TGpuBinarizedDataSet<TBinaryFeatureGridPolicy>& GetBinaryFeatures() const {
        return PermutationDataSets[0]->GetBinaryFeatures();
    }

    const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& GetFeatures() const {
        return PermutationDataSets[0]->GetFeatures();
    }

    const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& GetPermutationDependentFeatures(ui32 permutationId) const {
        return PermutationDataSets[permutationId]->GetTargetCtrs();
    }

    ui32 PermutationsCount() const {
        return (const ui32)PermutationDataSets.size();
    }

    const TDataPermutation& GetPermutation(ui32 permutationId) const {
        return PermutationDataSets[permutationId]->GetPermutation();
    }

    TDataSetsHolder() = default;

    TDataSetsHolder(const TDataProvider& dataProvider,
                    const TBinarizedFeaturesManager& featuresManager)
        : DataProvider(&dataProvider)
        , FeaturesManager(&featuresManager)
    {
    }

    bool HasTestDataSet() const {
        return TestDataSet != nullptr;
    }

    const TDataSet<CatFeaturesStoragePtrType>& GetTestDataSet() const {
        CB_ENSURE(HasTestDataSet());
        return *TestDataSet;
    }

    const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const {
        return *CtrTargets;
    }

    bool IsEmpty() const {
        return FeaturesManager == nullptr;
    }

private:
    const TDataProvider* DataProvider = nullptr;
    const TBinarizedFeaturesManager* FeaturesManager = nullptr;

    //learn target and weights
    TMirrorBuffer<float> DirectTarget;
    TMirrorBuffer<float> DirectWeights;
    //ctr_type
    TSimpleSharedPtr<TCtrTargets<NCudaLib::TMirrorMapping>> CtrTargets;

    yvector<TSimpleSharedPtr<TDataSet<CatFeaturesStoragePtrType>>> PermutationDataSets;
    THolder<TDataSet<CatFeaturesStoragePtrType>> TestDataSet;

    template <NCudaLib::EPtrType CatFeatureStoragePtrType>
    friend class TDataSetHoldersBuilder;
};
