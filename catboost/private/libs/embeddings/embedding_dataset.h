#pragma once

#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/ptr.h>


namespace NCB {
    using TEmbeddingsArray = TConstArrayRef<float>;
    using TEmbeddingArrayReferencesColumn = TMaybeOwningConstArrayHolder<TEmbeddingsArray>;

    class TEmbeddingDataSet : public TThrRefBase {
        TEmbeddingDataSet(TEmbeddingArrayReferencesColumn embed)
        : Embedding(std::move(embed)) {}

        ui64 SamplesCount() const {
            return (*Embedding).size();
        }

        ui64 GetDimention() const {
            CB_ENSURE(SamplesCount() > 0, "Error: empty embedding");
            return Embedding[0].size();
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

    struct TEmbeddingClassificationTarget : public TThrRefBase {
        TEmbeddingClassificationTarget(TVector<ui32>&& classes, ui32 numClasses)
        : Classes(std::move(classes))
        , NumClasses(numClasses)
        {}

    public:
        TVector<ui32> Classes;
        ui32 NumClasses;
    };

    using TEmbeddingDataSetPtr = TIntrusivePtr<TEmbeddingDataSet>;
    using TEmbeddingClassificationTargetPtr = TIntrusivePtr<TEmbeddingClassificationTarget>;
};
