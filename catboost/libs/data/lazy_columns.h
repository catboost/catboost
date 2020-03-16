#pragma once

#include "columns.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

namespace NCB {
    template <class T, EFeatureValuesType TType>
    class TLazyCompressedValuesHolderImpl : public TCloneableWithSubsetIndexingValuesHolder<T, TType> {
    public:
        using TBase = TCloneableWithSubsetIndexingValuesHolder<T, TType>;

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

        THolder<TCloneableWithSubsetIndexingValuesHolder<T, TType>> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TLazyCompressedValuesHolderImpl>(TBase::GetId(), subsetIndexing, PoolLoader);
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 /*offset*/) const override {
            // lazy compressed columns do not support iteration by blocks
            Y_UNREACHABLE();
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* /*localExecutor*/) const override {
            const ui32 featureId = TBase::GetId();
            return TMaybeOwningArrayHolder<T>::CreateOwning(PoolLoader->LoadQuantizedColumn(featureId));
        }

    private:
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
    };
}
