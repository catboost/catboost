#pragma once

#include "columns.h"

#include <catboost/libs/helpers/sparse_array.h>

namespace NCB {

    template <class TBase>
    class TSparsePolymorphicArrayValuesHolder: public TBase {
    public:
        using T = typename TBase::TValueType;
    public:
        TSparsePolymorphicArrayValuesHolder(ui32 featureId, TConstPolymorphicValuesSparseArray<T, ui32>&& data)
            : TBase(featureId, data.GetSize())
            , Data(std::move(data))
        {}

        bool IsSparse() const override {
            return true;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            return Data.EstimateGetSubsetCpuRamUsage(**cloningParams.InvertedSubsetIndexing);
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            return MakeHolder<TSparsePolymorphicArrayValuesHolder>(
                this->GetId(),
                this->GetData().GetSubset(**cloningParams.InvertedSubsetIndexing)
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::ILocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return TMaybeOwningArrayHolder<T>::CreateOwning(Data.ExtractValues());
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return Data.GetBlockIterator(offset);
        }

        const TConstPolymorphicValuesSparseArray<T, ui32>& GetData() const {
            return Data;
        }

    private:
        TConstPolymorphicValuesSparseArray<T, ui32> Data;
    };

    template <class TBase>
    class TSparseCompressedValuesHolderImpl : public TBase {
    public:
        TSparseCompressedValuesHolderImpl(ui32 featureId, TSparseCompressedArray<typename TBase::TValueType, ui32>&& data)
            : TBase(featureId, data.GetSize())
            , Data(std::move(data))
        {}

        bool IsSparse() const override {
            return true;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            return Data.EstimateGetSubsetCpuRamUsage(**cloningParams.InvertedSubsetIndexing);
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            return MakeHolder<TSparseCompressedValuesHolderImpl>(
                this->GetId(),
                this->GetData().GetSubset(**cloningParams.InvertedSubsetIndexing)
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            return Data.GetBlockIterator(offset);
        }

        const TSparseCompressedArray<typename TBase::TValueType, ui32>& GetData() const {
            return Data;
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            ui32 checkSum = 0;
            constexpr size_t BLOCK_SIZE = 10000;
            auto blockIterator = Data.GetBlockIterator();
            while (auto block = blockIterator->Next(BLOCK_SIZE)) {
                checkSum = UpdateCheckSum(checkSum, block);
            }
            return checkSum;
        }

    private:
        TSparseCompressedArray<typename TBase::TValueType, ui32> Data;
    };

    using TFloatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TFloatValuesHolder>;
    using THashedCatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<THashedCatValuesHolder>;
    using TStringTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TStringTextValuesHolder>;
    using TTokenizedTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TTokenizedTextValuesHolder>;

    using TQuantizedFloatSparseValuesHolder = TSparseCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatSparseValuesHolder = TSparseCompressedValuesHolderImpl<IQuantizedCatValuesHolder>;
}
