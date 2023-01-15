#pragma once

#include "columns.h"
#include "sparse_columns.h"
#include "quantized_features_info.h"
#include "sparse_columns.h"

#include <catboost/libs/helpers/sparse_array.h>

#include <util/system/compiler.h>
#include <util/system/yassert.h>


namespace NCB {
    class TExternalFloatValuesHolder: public IQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   ITypedArraySubsetPtr<float> srcData,
                                   TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedFloatValuesHolder(featureId, srcData->GetSize())
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

    private:
        ITypedArraySubsetPtr<float> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

    class TExternalCatValuesHolder: public IQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 ITypedArraySubsetPtr<ui32> srcData,
                                 TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedCatValuesHolder(featureId, srcData->GetSize())
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

    private:
        ITypedArraySubsetPtr<ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

    class TExternalFloatSparseValuesHolder : public IQuantizedFloatValuesHolder {
    public:
        TExternalFloatSparseValuesHolder(ui32 featureId,
                                         TConstPolymorphicValuesSparseArray<float, ui32> srcData,
                                         TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedFloatValuesHolder(
                featureId,
                srcData.GetSize()
              )
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return true;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override;

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

    private:
        TConstPolymorphicValuesSparseArray<float, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatSparseValuesHolder : public IQuantizedCatValuesHolder {
    public:
        TExternalCatSparseValuesHolder(ui32 featureId,
                                       TConstPolymorphicValuesSparseArray<ui32, ui32> srcData,
                                       TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedCatValuesHolder(featureId, srcData.GetSize())
            , SrcData(std::move(srcData))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return true;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override;

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

    private:
        TConstPolymorphicValuesSparseArray<ui32, ui32> SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}

