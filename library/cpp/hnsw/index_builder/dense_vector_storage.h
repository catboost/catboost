#pragma once

#include <util/generic/string.h>
#include <util/memory/blob.h>

namespace NHnsw {
    template <class TVectorComponent>
    class TDenseVectorStorage {
    public:
        using TItem = const TVectorComponent*;

        TDenseVectorStorage(const TString& vectorFilename, size_t dimension)
            : TDenseVectorStorage(TBlob::PrechargedFromFile(vectorFilename), dimension)
        {
        }

        TDenseVectorStorage(const TBlob& vectorData, size_t dimension)
            : Dimension(dimension)
            , VectorData(vectorData)
            , Vectors(reinterpret_cast<const TVectorComponent*>(VectorData.Begin()))
        {
        }

        size_t GetNumItems() const {
            return VectorData.Size() / sizeof(TVectorComponent) / Dimension;
        }
        const TVectorComponent* GetItem(size_t id) const {
            return Vectors + id * Dimension;
        }

        size_t GetDimension() const {
            return Dimension;
        }

    private:
        const size_t Dimension;
        TBlob VectorData;
        const TVectorComponent* Vectors;
    };

}
