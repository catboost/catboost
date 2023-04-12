#pragma once

#include "columns.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

namespace NCB {

    template <class TBase>
    class TLazyCompressedValuesHolderImpl : public TBase {
    public:
        TLazyCompressedValuesHolderImpl(ui32 featureId, const TPathWithScheme& pathWithScheme, ui64 size)
        : TBase(featureId, size)
        , PathWithScheme(pathWithScheme)
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

        ui32 CalcChecksum(NPar::ILocalExecutor* /*localExecutor*/) const override {
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* /*localExecutor*/
        ) const override {
            CB_ENSURE_INTERNAL(
                cloningParams.SubsetIndexing == nullptr || cloningParams.SubsetIndexing->IsFullSubset(),
                "Lazy columns support only full subset indexing");
            return MakeHolder<TLazyCompressedValuesHolderImpl>(
                TBase::GetId(),
                PathWithScheme,
                TBase::GetSize());
        }

        TPathWithScheme GetPathWithScheme() const {
            return PathWithScheme;
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 /*offset*/) const override {
            CB_ENSURE(false);
        }

    private:
        TPathWithScheme PathWithScheme;
    };

    using TLazyQuantizedFloatValuesHolder = TLazyCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;
    template <typename IQuantizedValuesHolder>
    const TLazyQuantizedFloatValuesHolder* CastToLazyQuantizedFloatValuesHolder(const IQuantizedValuesHolder* quantizedFeatureColumn);
}
