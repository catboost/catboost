#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/ptr.h>

namespace NCB {
    using TEmbeddingsArray = TMaybeOwningConstArrayHolder<float>;
    using TEmbeddingArrayReferencesColumn = TMaybeOwningConstArrayHolder<TEmbeddingsArray>;

    class TEmbeddingDataSet : public TThrRefBase {
    public:
        TEmbeddingDataSet(TEmbeddingArrayReferencesColumn embed)
        : Embedding(std::move(embed)) {}

        ui64 SamplesCount() const {
            return Embedding.GetSize();
        }

        ui64 GetDimension() const {
            CB_ENSURE(SamplesCount() > 0, "Error: empty embedding");
            return Embedding[0].GetSize();
        }

        const TEmbeddingsArray& GetVector(ui64 idx) const {
            const ui64 samplesCount = SamplesCount();
            CB_ENSURE(idx < samplesCount, "Error: text line " << idx << " is out of bound (" << samplesCount << ")");
            return Embedding[idx];
        }

        TConstArrayRef<TEmbeddingsArray> GetEmbedding() const {
            return *Embedding;
        }

    private:
        const TEmbeddingArrayReferencesColumn Embedding;
    };

    using TEmbeddingDataSetPtr = TIntrusivePtr<TEmbeddingDataSet>;
};
