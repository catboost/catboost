#pragma once

#include "dense_vector_storage.h"
#include <library/cpp/dot_product/dot_product.h>
#include <util/generic/buffer.h>
#include <util/memory/blob.h>
#include <util/system/yassert.h>

#include <type_traits>


namespace NHnsw {

/**
 * TransformMobius must provide one typedef:
 * - TItem
 * and three methods:
 * - GetItem(size_t id) - typically returning TItem or const TItem&
 * - size_t GetNumItems()
 * - size_t GetDimension()
 */

template <class TDenseVectorLikeStorage,
        class TVectorComponent = typename std::remove_const<typename std::remove_pointer<typename TDenseVectorLikeStorage::TItem>::type>::type,
        class TOutVectorComponent = typename std::conditional<std::is_same<TVectorComponent, double>::value, double, float>::type>
    TDenseVectorStorage<TOutVectorComponent> TransformMobius(const TDenseVectorLikeStorage& itemStorage) {
        if(itemStorage.GetNumItems() == 0)
            return TDenseVectorStorage<TOutVectorComponent>(TBlob(), itemStorage.GetDimension());
        TBuffer buffer;
        buffer.Resize(itemStorage.GetNumItems() * itemStorage.GetDimension() * sizeof(TOutVectorComponent));
        TOutVectorComponent* bufferPtr = reinterpret_cast<TOutVectorComponent*>(buffer.data());
        for(size_t i = 0; i < itemStorage.GetNumItems(); i++)
        {
            auto currentVector = itemStorage.GetItem(i);
            TOutVectorComponent vectorLengthSqr = DotProduct(currentVector, currentVector, itemStorage.GetDimension());
            Y_ABORT_UNLESS(vectorLengthSqr > 0, "All vectors should have positive length");
            for(size_t j = 0; j < itemStorage.GetDimension(); j++)
            {
                *bufferPtr = static_cast<TOutVectorComponent>(currentVector[j])/vectorLengthSqr;
                bufferPtr++;
            }
        }
        return TDenseVectorStorage<TOutVectorComponent>(TBlob::FromBuffer(buffer), itemStorage.GetDimension());
    }

}
