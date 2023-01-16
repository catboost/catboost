#pragma once

#include <util/generic/fwd.h>
#include <util/memory/blob.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#include <stddef.h>


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

        inline void PrefetchItem(size_t id) const {
            const auto& item = GetItem(id);
            const ui8* ptr = reinterpret_cast<const ui8*> (item);
            const ui8* ptr_end = reinterpret_cast<const ui8*> (item + Dimension);
            while (ptr < ptr_end) {
                Y_PREFETCH_READ(ptr, 1);
                ptr += 64u;
            }
            Y_PREFETCH_READ(ptr_end - 1, 1);
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
