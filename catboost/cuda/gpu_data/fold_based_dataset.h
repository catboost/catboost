#pragma once

#include "binarized_dataset.h"
#include "cat_features_dataset.h"
#include "ctr_helper.h"

#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/permutation.h>
namespace NCatboostCuda
{
    inline TCtrTargets<NCudaLib::TSingleMapping>
    DeviceView(const TCtrTargets<NCudaLib::TMirrorMapping>& mirrorTargets, ui32 devId)
    {
        TCtrTargets<NCudaLib::TSingleMapping> view;
        view.WeightedTarget = mirrorTargets.WeightedTarget.DeviceView(devId);
        view.BinarizedTarget = mirrorTargets.BinarizedTarget.DeviceView(devId);
        view.Weights = mirrorTargets.Weights.DeviceView(devId);

        view.TotalWeight = mirrorTargets.TotalWeight;
        view.LearnSlice = mirrorTargets.LearnSlice;
        view.TestSlice = mirrorTargets.TestSlice;
        return view;
    }

    template<class TLayoutPolicy = TCatBoostPoolLayout>
    class TGpuFeatures
    {
    public:
        const TGpuBinarizedDataSet<TBinaryFeatureGridPolicy, TLayoutPolicy>& GetBinaryFeatures() const
        {
            return BinaryFeatures;
        }

        const TGpuBinarizedDataSet<THalfByteFeatureGridPolicy, TLayoutPolicy>& GetHalfByteFeatures() const
        {
            return HalfByteFeatures;
        }

        const TGpuBinarizedDataSet<TByteFeatureGridPolicy, TLayoutPolicy>& GetFeatures() const
        {
            return Features;
        }

        bool HasFeature(ui32 featureId) const
        {
            if (GetBinaryFeatures().HasFeature(featureId))
            {
                return true;
            }
            if (GetHalfByteFeatures().HasFeature(featureId))
            {
                return true;
            }
            if (GetFeatures().HasFeature(featureId))
            {
                return true;
            }
            return false;
        }

        bool NotEmpty() const
        {
            return GetBinaryFeatures().NotEmpty() || GetHalfByteFeatures().NotEmpty() || GetFeatures().NotEmpty();
        }

        TVector<ui32> ComputeAllFeatureIds() const
        {
            TVector<ui32> result;
            const TVector<ui32>& binaryFeatureIds = GetBinaryFeatures().GetFeatureIds();
            result.insert(result.end(), binaryFeatureIds.begin(), binaryFeatureIds.end());
            const TVector<ui32>& halfByteFeatureIds = GetHalfByteFeatures().GetFeatureIds();
            result.insert(result.end(), halfByteFeatureIds.begin(), halfByteFeatureIds.end());
            const TVector<ui32>& byteFeatureIds = GetFeatures().GetFeatureIds();
            result.insert(result.end(), byteFeatureIds.begin(), byteFeatureIds.end());
            return result;
        }

        template<NCudaLib::EPtrType CatFeatureStoragePtrType>
        friend
        class TDataSetHoldersBuilder;

    private:
        TGpuBinarizedDataSet<TBinaryFeatureGridPolicy, TLayoutPolicy> BinaryFeatures;
        TGpuBinarizedDataSet<THalfByteFeatureGridPolicy, TLayoutPolicy> HalfByteFeatures;
        TGpuBinarizedDataSet<TByteFeatureGridPolicy, TLayoutPolicy> Features;
    };

    template<NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
    class TDataSet: public TGuidHolder
    {
    public:
        using TDocsMapping = NCudaLib::TMirrorMapping;

        static constexpr NCudaLib::EPtrType GetCatFeaturesStoragePtrType()
        {
            return CatFeaturesStoragePtrType;
        };

        TDataSet(const TDataProvider& dataProvider,
                 ui32 permutationId,
                 ui32 blockSize)
                : DataProvider(&dataProvider)
                  , Permutation(NCatboostCuda::GetPermutation(dataProvider, permutationId, blockSize))
        {
        }

        const TGpuFeatures<>& GetFeatures() const
        {
            CB_ENSURE(PermutationIndependentFeatures);
            return *PermutationIndependentFeatures;
        }

        const TGpuFeatures<>& GetPermutationFeatures() const
        {
            return PermutationDependentFeatures;
        }

        //target-ctr_type
        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const
        {
            CB_ENSURE(CatFeatures);
            return *CatFeatures;
        }

        bool HasFeature(ui32 featureId) const
        {
            if (PermutationIndependentFeatures->HasFeature(featureId))
            {
                return true;
            }
            return GetPermutationFeatures().HasFeature(featureId);
        }

        const TDataProvider& GetDataProvider() const
        {
            CB_ENSURE(DataProvider);
            return *DataProvider;
        }

        const TDataPermutation& GetPermutation() const
        {
            return Permutation;
        };

        const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const
        {
            return *CtrTargets;
        }

        bool HasCtrHistoryDataSet() const
        {
            return LinkedHistoryForCtrs != nullptr;
        }

        const TDataSet& LinkedHistoryForCtr() const
        {
            CB_ENSURE(HasCtrHistoryDataSet(), "No history dataset found");
            return *LinkedHistoryForCtrs;
        }

        //current permutation order
        const TMirrorBuffer<float>& GetTarget() const
        {
            return Targets;
        }

        //current permutation order
        const TMirrorBuffer<float>& GetWeights() const
        {
            return Weights;
        }

        //doc-indexing
        const TMirrorBuffer<ui32>& GetIndices() const
        {
            return Indices;
        }

        //doc-indexing
        const NCudaLib::TMirrorMapping& GetDocumentsMapping() const
        {
            return Targets.GetMapping();
        }

        //doc-indexing
        const TMirrorBuffer<ui32>& GetInverseIndices() const
        {
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

        TSimpleSharedPtr<TGpuFeatures<>> PermutationIndependentFeatures;
        TGpuFeatures<> PermutationDependentFeatures;

        TSimpleSharedPtr<TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>> CatFeatures;
        TSimpleSharedPtr<TCtrTargets<NCudaLib::TMirrorMapping>> CtrTargets;
        TSimpleSharedPtr<TDataSet> LinkedHistoryForCtrs;

        template<NCudaLib::EPtrType CatFeatureStoragePtrType>
        friend
        class TDataSetHoldersBuilder;
    };

    template<NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
    class TDataSetsHolder: public TGuidHolder
    {
    public:
        bool HasFeature(ui32 featureId) const
        {
            return PermutationDataSets.at(0)->HasFeature(featureId);
        }

        const TDataSet<CatFeaturesStoragePtrType>& GetDataSetForPermutation(ui32 permutationId) const
        {
            const auto* dataSetPtr = PermutationDataSets.at(permutationId).Get();
            CB_ENSURE(dataSetPtr);
            return *dataSetPtr;
        }

        const TDataProvider& GetDataProvider() const
        {
            CB_ENSURE(DataProvider);
            return *DataProvider;
        }

        const TBinarizedFeaturesManager& GetFeaturesManger() const
        {
            CB_ENSURE(FeaturesManager);
            return *FeaturesManager;
        }

        const TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& GetCatFeatures() const
        {
            return PermutationDataSets[0]->GetCatFeatures();
        }

        const TGpuFeatures<>& GetPermutationIndependentFeatures() const
        {
            CB_ENSURE(PermutationDataSets[0]);
            return PermutationDataSets[0]->GetFeatures();
        }

        const TGpuFeatures<>& GetPermutationDependentFeatures(ui32 permutationId) const
        {
            CB_ENSURE(PermutationDataSets[permutationId]);
            return PermutationDataSets[permutationId]->GetPermutationFeatures();
        }

        ui32 PermutationsCount() const
        {
            return (const ui32) PermutationDataSets.size();
        }

        const TDataPermutation& GetPermutation(ui32 permutationId) const
        {
            return PermutationDataSets[permutationId]->GetPermutation();
        }

        TDataSetsHolder() = default;

        TDataSetsHolder(const TDataProvider& dataProvider,
                        const TBinarizedFeaturesManager& featuresManager)
                : DataProvider(&dataProvider)
                  , FeaturesManager(&featuresManager)
        {
        }

        bool HasTestDataSet() const
        {
            return TestDataSet != nullptr;
        }

        const TDataSet<CatFeaturesStoragePtrType>& GetTestDataSet() const
        {
            CB_ENSURE(HasTestDataSet());
            return *TestDataSet;
        }

        const TCtrTargets<NCudaLib::TMirrorMapping>& GetCtrTargets() const
        {
            return *CtrTargets;
        }

        bool IsEmpty() const
        {
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

        TVector<TSimpleSharedPtr<TDataSet<CatFeaturesStoragePtrType>>> PermutationDataSets;
        THolder<TDataSet<CatFeaturesStoragePtrType>> TestDataSet;

        template<NCudaLib::EPtrType CatFeatureStoragePtrType>
        friend
        class TDataSetHoldersBuilder;
    };
}
