#pragma once

#include "columns.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

namespace NCB {

    template <class TBase>
    class TLazyCompressedValuesHolderImpl : public TBase {
    public:
        template<typename T>
        class TLazyCompressedValuesIterator : public IDynamicBlockIterator<T> {
        public:
            TLazyCompressedValuesIterator(TVector<T>&& values)
                : Values(std::move(values))
                , Iterator(Values)
            {

            }
            TConstArrayRef<T> Next(size_t blockSize) {
                return Iterator.Next(blockSize);
            }
        private:
            TVector<T> Values;
            TArrayBlockIterator<T> Iterator;
        };

        TLazyCompressedValuesHolderImpl(
            ui32 featureId,
            const TFeaturesArraySubsetIndexing* subsetIndexing,
            TAtomicSharedPtr<IQuantizedPoolLoader> poolLoader)
        : TBase(featureId, subsetIndexing->Size())
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

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(
                !cloningParams.MakeConsecutive,
                "Making TLazyCompressedValuesHolderImpl consecutive not supported for now"
            );
            CB_ENSURE_INTERNAL(
                cloningParams.SubsetIndexing->IsFullSubset(),
                "Making TLazyCompressedValuesHolderImpl with non full subset not supported for now"
            );
            return MakeHolder<TLazyCompressedValuesHolderImpl>(
                TBase::GetId(),
                cloningParams.SubsetIndexing,
                PoolLoader
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset) const override {
            CB_ENSURE_INTERNAL(offset == 0, "Non zero offset is not supported for now");
            const ui32 featureId = TBase::GetId();
            return MakeHolder<TLazyCompressedValuesIterator<ui8>>(
                PoolLoader->LoadQuantizedColumn(featureId)
            );
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(NPar::TLocalExecutor* /*localExecutor*/) const override {
            const ui32 featureId = TBase::GetId();
            return TMaybeOwningArrayHolder<typename TBase::TValueType>::CreateOwning(PoolLoader->LoadQuantizedColumn(featureId));
        }

    private:
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
    };
}
