#pragma once

#include <util/generic/vector.h>
#include <util/memory/blob.h>

namespace NOnlineHnsw {
    template <class TVectorComponent>
    class TDenseVectorExtendableItemStorage {
    public:
        using TItem = const TVectorComponent*;

        TDenseVectorExtendableItemStorage(size_t dimension, size_t maxSize = 0)
            : Dimension(dimension)
            , Size(0)
        {
            Data.reserve(maxSize * Dimension);
        }

        TDenseVectorExtendableItemStorage(TVector<TVectorComponent>&& data, size_t dimension, size_t size)
            : Dimension(dimension)
            , Data(data)
            , Size(size)
        {
        }

        TItem GetItem(ui32 id) const {
            return Data.data() + id * Dimension;
        }

        size_t GetDimension() const {
            return Dimension;
        }

        size_t GetNumItems() const {
            return Size;
        }

        void AddItem(const TItem item) {
            Data.insert(Data.end(), item, item + Dimension);
            ++Size;
        }

        const TVectorComponent* GetData() const {
            return Data.data();
        }

        size_t GetSize() const {
            return Data.size() * sizeof(TVectorComponent);
        }

        const TVector<TVectorComponent>& GetVector() const {
            return Data;
        }
    private:
        const size_t Dimension;
        TVector<TVectorComponent> Data;
        size_t Size;
    };
} // namespace NOnlineHnsw
