#pragma once

#include "columns.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

namespace NCB {

    template <class TBase>
    class TLazyCompressedValuesHolderImpl : public TBase {
    public:
        TLazyCompressedValuesHolderImpl(
            ui32 featureId,
            const TFeaturesArraySubsetIndexing* subsetIndexing,
            TAtomicSharedPtr<IQuantizedPoolLoader> poolLoader)
        : TBase(featureId, subsetIndexing->Size())
        , SubsetIndexing(subsetIndexing)
        , PoolLoader(poolLoader)
        {
        }

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(!cloningParams.MakeConsecutive, "Making consecutive not supported on Lazy columns for now");
            return MakeHolder<TLazyCompressedValuesHolderImpl>(
                TBase::GetId(),
                cloningParams.SubsetIndexing,
                PoolLoader
            );
        }

        TPathWithScheme GetPoolPathWithScheme() const {
            return PoolLoader->GetPoolPathWithScheme();
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 /*offset*/) const override {
            CB_ENSURE(false);
        }

    private:
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
    };

    using TLazyQuantizedFloatValuesHolder = TLazyCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;
    template <typename IQuantizedValuesHolder>
    const TLazyQuantizedFloatValuesHolder* CastToLazyQuantizedFloatValuesHolder(const IQuantizedValuesHolder* quantizedFeatureColumn);
}
