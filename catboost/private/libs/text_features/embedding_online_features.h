#pragma once

#include "feature_calcer.h"

#include <catboost/private/libs/text_processing/embedding.h>

#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/array_ref.h>

namespace NCB {

    class TEmbeddingOnlineFeatures final : public TTextFeatureCalcer {
    public:

        explicit TEmbeddingOnlineFeatures(
            const TGuid& calcerId = CreateGuid(),
            ui32 numClasses = 2,
            TEmbeddingPtr embedding = TEmbeddingPtr(),
            bool useCos = true,
            bool computeHomoscedasticModel = true,
            bool computeHeteroscedasticModel = true,
            double prior = 1
        )
            : TTextFeatureCalcer(
                BaseFeatureCount(
                    numClasses,
                    useCos,
                    computeHomoscedasticModel,
                    computeHeteroscedasticModel
                ),
                calcerId
            )
            , NumClasses(numClasses)
            , Embedding(std::move(embedding))
            , ComputeCosDistance(useCos)
            , ComputeHomoscedasticModel(computeHomoscedasticModel)
            , ComputeHeteroscedasticModel(computeHeteroscedasticModel)
            , Prior(prior)
            , TotalWeight(prior)
            , Means(numClasses)
            , PerClassSigma(numClasses)
            , ClassSizes(numClasses)
        {
            const auto embeddingsDim = Embedding->Dim();

            TotalSigma = TVector<double>(embeddingsDim * embeddingsDim);

            for (auto& vec : Means) {
                vec.resize(embeddingsDim);
            }
            for (auto& vec : PerClassSigma) {
                vec.resize(embeddingsDim * embeddingsDim);
            }
        }

        static ui32 BaseFeatureCount(
            ui32 numClasses,
            bool computeCosDistance,
            bool computeHomoscedasticModel,
            bool computeHeteroscedasticModel
        ) {
            return numClasses * (
                (ui32)(computeCosDistance) +
                (ui32)(computeHomoscedasticModel) +
                (ui32)(computeHeteroscedasticModel)
            );
        }

        void Compute(TConstArrayRef<float> embedding, TOutputFloatIterator outputFeaturesIterator) const;

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::EmbeddingDistanceToClass;
        }

        void Compute(const TText& text, TOutputFloatIterator iterator) const override {
            TVector<float> embedding;
            Embedding->Apply(text, &embedding);
            return Compute(embedding, iterator);
        }

        void SetEmbedding(TEmbeddingPtr embedding) {
            Embedding = std::move(embedding);
        }

        bool IsSerializable() const override {
            return true;
        }

    private:
        ui32 NumClasses;
        TEmbeddingPtr Embedding;

        bool ComputeCosDistance;
        bool ComputeHomoscedasticModel;
        bool ComputeHeteroscedasticModel;
        double Prior;
        double TotalWeight;

        TVector<double> TotalSigma;
        TVector<TVector<double>> Means;
        TVector<TVector<double>> PerClassSigma;
        TVector<ui64> ClassSizes;

        friend class TEmbeddingFeaturesVisitor;
    };

    class TEmbeddingFeaturesVisitor final : public ITextCalcerVisitor {
    public:
        TEmbeddingFeaturesVisitor(ui32 numClasses, ui32 embeddingsDim)
        : NumClasses(numClasses)
        , Dim(embeddingsDim)
        , Sums(numClasses)
        , Sums2(numClasses) {
            for (auto& vec : Sums) {
                vec.resize(Dim);
            }
            for (auto& vec : Sums2) {
                vec.resize(Dim * (Dim + 1) / 2);
            }
        }

        void Update(ui32 classIdx, const TText& text, TTextFeatureCalcer* calcer) override {
            auto embeddingCalcer = dynamic_cast<TEmbeddingOnlineFeatures*>(calcer);
            Y_ASSERT(embeddingCalcer);

            TVector<float> embedding;
            embeddingCalcer->Embedding->Apply(text, &embedding);
            UpdateEmbedding(classIdx, embedding, embeddingCalcer);
        }

        void UpdateEmbedding(
            ui32 classId,
            TConstArrayRef<float> embedding,
            TEmbeddingOnlineFeatures* embeddingCalcer
        );

    private:
        const ui32 NumClasses;
        const ui32 Dim;
        TVector<TVector<double>> Sums;
        TVector<TVector<double>> Sums2;
    };
}
