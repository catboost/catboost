#pragma once

#include <util/generic/vector.h>
#include <util/memory/blob.h>

namespace NHnsw {

    template <class TVectorComponent>
    class TDenseVectorItemStorage {
    public:
        using TItem = const TVectorComponent*;

        TDenseVectorItemStorage(const TBlob& vectorData, size_t dimension)
            : Dimension(dimension)
            , VectorData(vectorData)
            , Vectors(reinterpret_cast<const TVectorComponent*>(VectorData.Begin()))
        {
        }

        const TVectorComponent* GetItem(ui32 id) const {
            return Vectors + id * Dimension;
        }

        size_t GetDimension() const {
            return Dimension;
        }

        size_t GetNumItems() const {
            return VectorData.Size() / sizeof(TVectorComponent) / Dimension;
        }

        size_t GetDataSize() const {
            return VectorData.Size() / sizeof(TVectorComponent);
        }

    private:
        const size_t Dimension;
        TBlob VectorData;
        const TVectorComponent* Vectors;
    };

}
