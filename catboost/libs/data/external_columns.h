#pragma once

#include "columns.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/sparse_array.h>

#include <util/system/compiler.h>
#include <util/system/yassert.h>


namespace NCB {

    class TExternalFloatValuesHolder: public ICloneableQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   ITypedArraySubsetPtr<float> srcData,
                                   TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedFloatValuesHolder(featureId, srcData->GetSize())
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        THolder<ICloneableQuantizedFloatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

        IDynamicBlockIteratorPtr<ui8> GetBlockIterator(ui32 offset = 0) const override {
            Y_UNUSED(offset);
            /* TODO(akhropov): Implement block iterators for external columns.
             *  Not currently used as external columns are used only in GPU training
             */
            Y_FAIL("GetBlockIterator unimplemented for external columns");
            Y_UNREACHABLE();
        }

    private:
        ITypedArraySubsetPtr<float> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatValuesHolder: public ICloneableQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 ITypedArraySubsetPtr<ui32> srcData,
                                 TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedCatValuesHolder(featureId, srcData->GetSize())
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        THolder<ICloneableQuantizedCatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

        IDynamicBlockIteratorPtr<ui32> GetBlockIterator(ui32 offset = 0) const override {
            Y_UNUSED(offset);
            /* TODO(akhropov): Implement block iterators for external columns.
             *  Not currently used as external columns are used only in GPU training
             */
            Y_FAIL("GetBlockIterator unimplemented for external columns");
            Y_UNREACHABLE();
        }

    private:
        ITypedArraySubsetPtr<ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

    class TExternalFloatSparseValuesHolder
        : public TValuesHolderWithScheduleGetSubset<ui8, EFeatureValuesType::QuantizedFloat> {
    public:
        TExternalFloatSparseValuesHolder(ui32 featureId,
                                         TConstPolymorphicValuesSparseArray<float, ui32> srcData,
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

        IDynamicBlockIteratorPtr<ui8> GetBlockIterator(ui32 offset = 0) const override {
            Y_UNUSED(offset);
            /* TODO(akhropov): Implement block iterators for external columns.
            *  Not currently used as external columns are used only in GPU training
            */
            Y_FAIL("GetBlockIterator unimplemented for external columns");
            Y_UNREACHABLE();
        }

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
        TConstPolymorphicValuesSparseArray<float, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatSparseValuesHolder
        : public TValuesHolderWithScheduleGetSubset<ui32, EFeatureValuesType::PerfectHashedCategorical> {
    public:
        TExternalCatSparseValuesHolder(ui32 featureId,
                                       TConstPolymorphicValuesSparseArray<ui32, ui32> srcData,
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

        IDynamicBlockIteratorPtr<ui32> GetBlockIterator(ui32 offset = 0) const override {
            Y_UNUSED(offset);
            /* TODO(akhropov): Implement block iterators for external columns.
            *  Not currently used as external columns are used only in GPU training
            */
            Y_FAIL("GetBlockIterator unimplemented for external columns");
            Y_UNREACHABLE();
        }

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
        TConstPolymorphicValuesSparseArray<ui32, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}

