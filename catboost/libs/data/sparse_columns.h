#pragma once

#include "columns.h"

#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/helpers/sparse_array.h>


namespace NCB {
    using TFeaturesSparseArrayIndexing = TSparseArrayIndexing<ui32>;

    template <class T, EFeatureValuesType TType>
    struct TValuesHolderWithScheduleGetSubset : public TTypedFeatureValuesHolder<T, TType>
    {
        TValuesHolderWithScheduleGetSubset(ui32 featureId, ui32 size, bool isSparse)
            : TTypedFeatureValuesHolder<T, TType>(featureId, size, isSparse)
        {}

        /* getting subset might require additional data, so use TResourceConstrainedExecutor
         */
        virtual void ScheduleGetSubset(
            // pointer to capture in lambda
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
        ) const = 0;
    };

    template <class T, EFeatureValuesType TType>
    class TSparsePolymorphicArrayValuesHolder: public TValuesHolderWithScheduleGetSubset<T, TType> {
    public:
        TSparsePolymorphicArrayValuesHolder(ui32 featureId, TConstPolymorphicValuesSparseArray<T, ui32>&& data)
            : TValuesHolderWithScheduleGetSubset<T, TType>(featureId, data.GetSize(), /*isSparse*/ true)
            , Data(std::move(data))
        {}

        void ScheduleGetSubset(
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
        ) const override {
            resourceConstrainedExecutor->Add(
                {
                    Data.EstimateGetSubsetCpuRamUsage(*subsetInvertedIndexing),
                    [this, subsetInvertedIndexing, subsetDst] () {
                        *subsetDst = MakeHolder<TSparsePolymorphicArrayValuesHolder>(
                            this->GetId(),
                            this->GetData().GetSubset(*subsetInvertedIndexing)
                        );
                    }
                }
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return TMaybeOwningArrayHolder<T>::CreateOwning(Data.ExtractValues());
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return MakeHolder<typename TConstPolymorphicValuesSparseArray<T, ui32>::TBlockIterator>(
                Data.GetBlockIterator(offset)
            );
        }

        const TConstPolymorphicValuesSparseArray<T, ui32>& GetData() const {
            return Data;
        }

    private:
        TConstPolymorphicValuesSparseArray<T, ui32> Data;
    };
    using TFloatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<float, EFeatureValuesType::Float>;
    using THashedCatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using TStringTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TString, EFeatureValuesType::StringText>;


    template <class T, EFeatureValuesType TType>
    class TSparseCompressedValuesHolderImpl : public TValuesHolderWithScheduleGetSubset<T, TType> {
    public:
        using TBase = TValuesHolderWithScheduleGetSubset<T, TType>;

    public:
        TSparseCompressedValuesHolderImpl(ui32 featureId, TSparseCompressedArray<T, ui32>&& data)
            : TBase(featureId, data.GetSize(), /*isSparse*/ true)
            , Data(std::move(data))
        {}

        void ScheduleGetSubset(
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
        ) const override {
            resourceConstrainedExecutor->Add(
                {
                    Data.EstimateGetSubsetCpuRamUsage(*subsetInvertedIndexing),
                    [this, subsetInvertedIndexing, subsetDst] () {
                        *subsetDst = MakeHolder<TSparseCompressedValuesHolderImpl<T, TType>>(
                            this->GetId(),
                            this->GetData().GetSubset(*subsetInvertedIndexing)
                        );
                    }
                }
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return TMaybeOwningArrayHolder<T>::CreateOwning(Data.ExtractValues());
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return MakeHolder<typename TSparseCompressedArray<T, ui32>::TBlockIterator>(
                Data.GetBlockIterator(offset)
            );
        }

        const TSparseCompressedArray<T, ui32>& GetData() const {
            return Data;
        }

    private:
        TSparseCompressedArray<T, ui32> Data;
    };

    using TQuantizedFloatSparseValuesHolder
        = TSparseCompressedValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;

    using TQuantizedCatSparseValuesHolder
        = TSparseCompressedValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using TTokenizedTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TText, EFeatureValuesType::TokenizedText>;

}
