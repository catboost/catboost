#pragma once

#include "embedding.h"
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/array_ref.h>

namespace NCB {

    class TEmbeddingOnlineFeatures {
    public:

        explicit TEmbeddingOnlineFeatures(ui32 numClasses,
                                          TEmbeddingPtr embedding,
                                          bool useCos = true,
                                          bool computeHomoscedasticModel = true,
                                          bool computeHeteroscedasticModel = true,
                                          double prior = 1
                                          )
            : NumClasses(numClasses)
            , Embedding(std::move(embedding))
            , UseCos(useCos)
            , ComputeHomoscedasticModel(computeHomoscedasticModel)
            , ComputeHeteroscedasticModel(computeHeteroscedasticModel)
            , Prior(prior)
            , Sums(numClasses)
            , Sums2(numClasses)
            , ClassSizes(numClasses) {
            const auto embeddingsDim = Embedding->Dim();

            for (auto& vec : Sums) {
                vec.resize(embeddingsDim);
            }
            for (auto& vec : Sums2) {
                vec.resize(embeddingsDim * (embeddingsDim + 1) / 2);
            }
        }

        TVector<double> CalcFeatures(TConstArrayRef<float> embedding) const;

        TVector<double> CalcFeaturesAndAddEmbedding(ui32 classId, TConstArrayRef<float> embedding);

        void AddEmbedding(ui32 classId, TConstArrayRef<float> embedding);

        TVector<double> CalcFeatures(const TText& text) const {
            TVector<float> embedding;
            Embedding->Apply(text, &embedding);
            return CalcFeatures(embedding);
        }

        TVector<double> CalcFeaturesAndAddText(ui32 classId, const TText& text) {
            TVector<float> embedding;
            Embedding->Apply(text, &embedding);
            return CalcFeaturesAndAddEmbedding(classId, embedding);
        }

        void AddText(ui32 classId, const TText& text) {
            TVector<float> embedding;
            Embedding->Apply(text, &embedding);
            AddEmbedding(classId, embedding);
        }

    private:
        ui32 NumClasses;
        TEmbeddingPtr Embedding;

        bool UseCos;
        bool ComputeHomoscedasticModel;
        bool ComputeHeteroscedasticModel;
        double Prior;

        TVector<TVector<double>> Sums;
        TVector<TVector<double>> Sums2;
        TVector<ui64> ClassSizes;
    };

}
