#pragma once

#include <catboost/libs/data/objects.h>

#include <catboost/libs/data/columns.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <numeric>


namespace NCB {
    namespace NDataNewUT {

    template <typename T, class TColumn>
    void InitFeatures(
        const TVector<TVector<T>>& src,
        const TArraySubsetIndexing<ui32>& indexing,
        TConstArrayRef<ui32> featureIds,
        TVector<THolder<TColumn>>* dst
    ) {
        using TValueType = typename TColumn::TValueType;
        for (auto i : xrange(src.size())) {
            dst->emplace_back(
                MakeHolder<TPolymorphicArrayValuesHolder<TColumn>>(
                    featureIds[i],
                    TMaybeOwningConstArrayHolder<TValueType>::CreateOwning( TVector<TValueType>(src[i]) ),
                    &indexing
                )
            );
        }
    }

    template <typename T, class TColumn>
    void InitFeatures(
        const TVector<TVector<T>>& src,
        const TArraySubsetIndexing<ui32>& indexing,
        ui32* featureId,
        TVector<THolder<TColumn>>* dst
    ) {
        TVector<ui32> featureIds(src.size());
        std::iota(featureIds.begin(), featureIds.end(), *featureId);
        InitFeatures(src, indexing, featureIds, dst);
        *featureId += (ui32)src.size();
    }

    template <typename T, class TColumn>
    void InitQuantizedFeatures(
        const TVector<TVector<T>>& src,
        const TFeaturesArraySubsetIndexing* subsetIndexing,
        TConstArrayRef<ui32> featureIds,
        TVector<THolder<TColumn>>* dst
    ) {
        dst->clear();
        for (auto perTypeFeatureIdx : xrange(src.size())) {
            const auto& srcColumn = src[perTypeFeatureIdx];
            ui32 bitsPerKey = sizeof(T)*8;
            auto storage = TMaybeOwningArrayHolder<ui64>::CreateOwning(
                CompressVector<ui64>(srcColumn.data(), srcColumn.size(), bitsPerKey)
            );

            dst->emplace_back(
                MakeHolder<TCompressedValuesHolderImpl<TColumn>>(
                    featureIds[perTypeFeatureIdx],
                    TCompressedArray(srcColumn.size(), bitsPerKey, storage),
                    subsetIndexing
                )
            );
        }
    }


    void Compare(const TQuantizedObjectsDataProvider& lhs, const TQuantizedObjectsDataProvider& rhs);

    }

}
