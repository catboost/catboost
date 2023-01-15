#pragma once

#include "feature_index.h"
#include "meta_info.h"
#include "objects.h"
#include "unaligned_mem.h"
#include "util.h"

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <catboost/libs/helpers/sparse_array.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/quantization_schema/schema.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {

    enum class EDatasetVisitorType {
        RawObjectsOrder,
        RawFeaturesOrder,
        QuantizedFeatures
    };

    // methods common to all implementations
    class IDatasetVisitor {
    public:
        virtual ~IDatasetVisitor() = default;

        virtual EDatasetVisitorType GetType() const = 0;

        // separate method because they can be loaded from a separate data source
        virtual void SetGroupWeights(TVector<float>&& groupWeights) = 0;

        // separate method because they can be loaded from a separate data source
        virtual void SetBaseline(TVector<TVector<float>>&& baseline) = 0;

        virtual void SetPairs(TVector<TPair>&& pairs) = 0;

        virtual void SetTimestamps(TVector<ui64>&& timestamps) = 0;

        // less effective version for Cython
        void SetPairs(TConstArrayRef<TPair> pairs) {
            TVector<TPair> pairsCopy;
            Assign(pairs, &pairsCopy);
            SetPairs(std::move(pairsCopy));
        }

        /* needed for checking groupWeights consistency while loading from separate file
         *  returns Nothing() if visitor has no GroupIds
         *
         * but getting something from possibly streaming visitor is wrong and incompatible with blocked
         * processing, so this should be fixed: TODO(akhropov): MLTOOLS-2358.
         */
        virtual TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const = 0;
    };


    class IRawObjectsOrderDataVisitor : public IDatasetVisitor {
    public:
        constexpr static EDatasetVisitorType Type = EDatasetVisitorType::RawObjectsOrder;

    public:
        EDatasetVisitorType GetType() const override {
            return Type;
        }

        virtual void Start(
            bool inBlock, // subset processing - Start/Finish is called for each block
            const TDataMetaInfo& metaInfo,
            bool haveUnknownNumberOfSparseFeatures,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders
        ) = 0;

        virtual void StartNextBlock(ui32 blockSize) = 0;

        // TCommonObjectsData
        virtual void AddGroupId(ui32 localObjectIdx, TGroupId value) = 0;
        virtual void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) = 0;
        virtual void AddTimestamp(ui32 localObjectIdx, ui64 value) = 0;

        // TRawObjectsData
        virtual void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) = 0;
        virtual void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) = 0;
        virtual void AddAllFloatFeatures(
            ui32 localObjectIdx,
            TConstPolymorphicValuesSparseArray<float, ui32> features
        ) = 0;

        // for sparse float features default value is always assumed to be 0.0f

        virtual ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) = 0;
        // localObjectIdx may be used as hint for sampling
        virtual ui32 GetCatFeatureValue(ui32 /* localObjectIdx */, ui32 flatFeatureIdx, TStringBuf feature) {
            return GetCatFeatureValue(flatFeatureIdx, feature);
        }
        virtual void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) = 0;
        virtual void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) = 0;
        virtual void AddAllCatFeatures(
            ui32 localObjectIdx,
            TConstPolymorphicValuesSparseArray<ui32, ui32> features
        ) = 0;

        // for sparse data
        virtual void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) = 0;

        virtual void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) = 0;
        virtual void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) = 0;
        virtual void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) = 0;
        virtual void AddAllTextFeatures(
            ui32 localObjectIdx,
            TConstPolymorphicValuesSparseArray<TString, ui32> features
        ) = 0;

        // TRawTargetData

        virtual void AddTarget(ui32 localObjectIdx, const TString& value) = 0;
        virtual void AddTarget(ui32 localObjectIdx, float value) = 0;
        virtual void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) = 0;
        virtual void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) = 0;
        virtual void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) = 0;
        virtual void AddWeight(ui32 localObjectIdx, float value) = 0;
        virtual void AddGroupWeight(ui32 localObjectIdx, float value) = 0;

        virtual void Finish() = 0;

    };


    class IRawFeaturesOrderDataVisitor : public IDatasetVisitor {
    public:
        constexpr static EDatasetVisitorType Type = EDatasetVisitorType::RawFeaturesOrder;

    public:
        EDatasetVisitorType GetType() const override {
            return Type;
        }

        virtual void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders
        ) = 0;

        // TCommonObjectsData
        virtual void AddGroupId(ui32 objectIdx, TGroupId value) = 0;
        virtual void AddSubgroupId(ui32 objectIdx, TSubgroupId value) = 0;
        virtual void AddTimestamp(ui32 objectIdx, ui64 value) = 0;

        // TRawObjectsData

        // shared ownership is passed to IRawFeaturesOrderDataVisitor
        virtual void AddFloatFeature(ui32 flatFeatureIdx, ITypedSequencePtr<float> features) = 0;
        virtual void AddFloatFeature(
            ui32 flatFeatureIdx,
            TConstPolymorphicValuesSparseArray<float, ui32> features
        ) = 0;

        virtual ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) = 0;
        virtual void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TString> feature) = 0;
        virtual void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef<TStringBuf> feature) = 0;
        virtual void AddCatFeature(
            ui32 flatFeatureIdx,
            TConstPolymorphicValuesSparseArray<TString, ui32> features
        ) = 0;

        // when hashes already computed
        // shared ownership is passed to IRawFeaturesOrderDataVisitor
        virtual void AddCatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<ui32> features) = 0;

        virtual void AddTextFeature(ui32 flatFeatureIdx, TConstArrayRef<TString> features) = 0;
        virtual void AddTextFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder<TString> feature) = 0;
        virtual void AddTextFeature(
            ui32 flatFeatureIdx,
            TConstPolymorphicValuesSparseArray<TString, ui32> features
        ) = 0;

        // TRawTargetData

        virtual void AddTarget(TConstArrayRef<TString> value) = 0;
        virtual void AddTarget(ITypedSequencePtr<float> value) = 0;
        virtual void AddTarget(ui32 flatTargetIdx, TConstArrayRef<TString> value) = 0;
        virtual void AddTarget(ui32 flatTargetIdx, ITypedSequencePtr<float> value) = 0;
        virtual void AddBaseline(ui32 baselineIdx, TConstArrayRef<float> value) = 0;
        virtual void AddWeights(TConstArrayRef<float> value) = 0;
        virtual void AddGroupWeights(TConstArrayRef<float> value) = 0;

        virtual void Finish() = 0;

    };


    class IQuantizedFeaturesDataVisitor : public IDatasetVisitor {
    public:
        constexpr static EDatasetVisitorType Type = EDatasetVisitorType::QuantizedFeatures;

    public:
        EDatasetVisitorType GetType() const override {
            return Type;
        }

        virtual ~IQuantizedFeaturesDataVisitor() = default;

        virtual void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders,

            const NCB::TPoolQuantizationSchema& poolQuantizationSchema
        ) = 0;

        // TCommonObjectsData
        virtual void AddGroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TGroupId> groupIdPart) = 0;
        virtual void AddSubgroupIdPart(ui32 objectOffset, TUnalignedArrayBuf<TSubgroupId> subgroupIdPart) = 0;
        virtual void AddTimestampPart(ui32 objectOffset, TUnalignedArrayBuf<ui64> timestampPart) = 0;


        // TQuantizedObjectsData

        /* shared ownership is passed to Start in resourceHolders to avoid creating resource holder
         * for each such call
         */
        virtual void AddFloatFeaturePart(
            ui32 flatFeatureIdx,
            ui32 objectOffset,
            ui8 bitsPerDocumentFeature,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-object data size depends on BitsPerKey
        ) = 0;

        virtual void AddCatFeaturePart(
            ui32 flatFeatureIdx,
            ui32 objectOffset,
            ui8 bitsPerDocumentFeature,
            TMaybeOwningConstArrayHolder<ui8> featuresPart // per-object data size depends on BitsPerKey
        ) = 0;


        // TRawTargetData

        virtual void AddTargetPart(ui32 objectOffset, TUnalignedArrayBuf<float> targetPart) = 0;
        virtual void AddTargetPart(ui32 objectOffset, TMaybeOwningConstArrayHolder<TString> targetPart) = 0;
        virtual void AddTargetPart(ui32 flatTargetIdx, ui32 objectOffset, TUnalignedArrayBuf<float> targetPart) = 0;
        virtual void AddTargetPart(ui32 flatTargetIdx, ui32 objectOffset, TMaybeOwningConstArrayHolder<TString> targetPart) = 0;

        virtual void AddBaselinePart(
            ui32 objectOffset,
            ui32 baselineIdx,
            TUnalignedArrayBuf<float> baselinePart
        ) = 0;
        virtual void AddWeightPart(ui32 objectOffset, TUnalignedArrayBuf<float> weightPart) = 0;
        virtual void AddGroupWeightPart(ui32 objectOffset, TUnalignedArrayBuf<float> groupWeightPart) = 0;

        virtual void Finish() = 0;

    };

}
