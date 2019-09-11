#pragma once

#include <catboost/libs/data_new/objects.h>

#include <catboost/libs/data_new/columns.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <numeric>


namespace NCB {
    namespace NDataNewUT {

    template <class T, EFeatureValuesType TType>
    void InitFeatures(
        const TVector<TVector<T>>& src,
        const TArraySubsetIndexing<ui32>& indexing,
        TConstArrayRef<ui32> featureIds,
        TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>* dst
    ) {
        for (auto i : xrange(src.size())) {
            dst->emplace_back(
                MakeHolder<TPolymorphicArrayValuesHolder<T, TType>>(
                    featureIds[i],
                    TMaybeOwningConstArrayHolder<T>::CreateOwning( TVector<T>(src[i]) ),
                    &indexing
                )
            );
        }
    }

    template <class T, EFeatureValuesType TType>
    void InitFeatures(
        const TVector<TVector<T>>& src,
        const TArraySubsetIndexing<ui32>& indexing,
        ui32* featureId,
        TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>* dst
    ) {
        TVector<ui32> featureIds(src.size());
        std::iota(featureIds.begin(), featureIds.end(), *featureId);
        InitFeatures(src, indexing, featureIds, dst);
        *featureId += (ui32)src.size();
    }

    template <class T, class TColumnData, EFeatureValuesType FeatureValuesType>
    void InitQuantizedFeatures(
        const TVector<TVector<T>>& src,
        const TFeaturesArraySubsetIndexing* subsetIndexing,
        TConstArrayRef<ui32> featureIds,
        TVector<THolder<TTypedFeatureValuesHolder<TColumnData, FeatureValuesType>>>* dst
    ) {
        dst->clear();
        for (auto perTypeFeatureIdx : xrange(src.size())) {
            const auto& srcColumn = src[perTypeFeatureIdx];
            ui32 bitsPerKey = sizeof(T)*8;
            auto storage = TMaybeOwningArrayHolder<ui64>::CreateOwning(
                CompressVector<ui64>(srcColumn.data(), srcColumn.size(), bitsPerKey)
            );

            dst->emplace_back(
                MakeHolder<TCompressedValuesHolderImpl<TColumnData, FeatureValuesType>>(
                    featureIds[perTypeFeatureIdx],
                    TCompressedArray(srcColumn.size(), bitsPerKey, storage),
                    subsetIndexing
                )
            );
        }
    }


    void Compare(const TQuantizedObjectsDataProvider& lhs, const TQuantizedObjectsDataProvider& rhs);

    void Compare(
        const TQuantizedForCPUObjectsDataProvider& lhs,
        const TQuantizedForCPUObjectsDataProvider& rhs
    );

    }

}
