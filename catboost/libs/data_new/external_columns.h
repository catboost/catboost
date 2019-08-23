#pragma once

#include "columns.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/sparse_array.h>


namespace NCB {

    class TExternalFloatValuesHolder: public ICloneableQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   NCB::TMaybeOwningConstArrayHolder<float> srcData,
                                   const TFeaturesArraySubsetIndexing* subsetIndexing,
                                   TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedFloatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<ICloneableQuantizedFloatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningConstArrayHolder<float> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatValuesHolder: public ICloneableQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 NCB::TMaybeOwningConstArrayHolder<ui32> srcData,
                                 const TFeaturesArraySubsetIndexing* subsetIndexing,
                                 TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedCatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<ICloneableQuantizedCatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningConstArrayHolder<ui32> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

    class TExternalFloatSparseValuesHolder
        : public TValuesHolderWithScheduleGetSubset<ui8, EFeatureValuesType::QuantizedFloat> {
    public:
        TExternalFloatSparseValuesHolder(ui32 featureId,
                                         TConstSparseArray<float, ui32> srcData,
                                         TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : TValuesHolderWithScheduleGetSubset<ui8, EFeatureValuesType::QuantizedFloat>(
                featureId,
                srcData.GetSize(),
                /*isSparse*/ true
              )
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

        void ScheduleGetSubset(
            // pointer to capture in lambda
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<IQuantizedFloatValuesHolder>* subsetDst
        ) const override;

        THolder<TExternalFloatSparseValuesHolder> Clone() const {
            return MakeHolder<TExternalFloatSparseValuesHolder>(GetId(), SrcData, QuantizedFeaturesInfo);
        }

    private:
        TConstSparseArray<float, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatSparseValuesHolder
        : public TValuesHolderWithScheduleGetSubset<ui32, EFeatureValuesType::PerfectHashedCategorical> {
    public:
        TExternalCatSparseValuesHolder(ui32 featureId,
                                       TConstSparseArray<ui32, ui32> srcData,
                                       TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : TValuesHolderWithScheduleGetSubset<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                featureId,
                srcData.GetSize(),
                /*isSparse*/ true
              )
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

        void ScheduleGetSubset(
            // pointer to capture in lambda
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<IQuantizedCatValuesHolder>* subsetDst
        ) const override;

        THolder<TExternalCatSparseValuesHolder> Clone() const {
            return MakeHolder<TExternalCatSparseValuesHolder>(GetId(), SrcData, QuantizedFeaturesInfo);
        }

    private:
        TConstSparseArray<ui32, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}

