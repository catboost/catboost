#pragma once

#include "embedding_feature_calcer.h"

namespace NCB {

    class IncrementalCloud {
    public:
        IncrementalCloud(int dim)
            : Dimension(dim)
            , BaseCenter(dim, 0)
            , NewShift(dim, 0)
            , ScatterMatrix(dim * dim, 0)
        {}
        void AddVector(const TEmbeddingsArray& embed);
        void Update();

        float TotalSize() {
            return BaseSize + AdditionalSize;
        }
    public:
        int Dimension;
        int BaseSize = 0;
        int AdditionalSize = 0;
        TVector<float> BaseCenter;
        TVector<float> NewShift;
        TVector<float> ScatterMatrix;
        TVector<float> Buffer;
    };

    class TLinearDACalcer final : public TEmbeddingFeatureCalcer {
    public:
        explicit TLinearDACalcer(
            int totalDimension,
            int numClasses,
            int projectionDimension,
            float regularization = 0.01,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(projectionDimension, calcerId)
            , TotalDimension(totalDimension)
            , NumClasses(numClasses)
            , ProjectionDimension(projectionDimension)
            , RegParam(regularization)
            , ClassesDist(numClasses, totalDimension)
            , ProjectionMatrix(totalDimension * projectionDimension)
            , EigenValues(TotalDimension)
            , ProjectionCalculationCache(totalDimension * (totalDimension + 2))
        {}

        void Compute(const TEmbeddingsArray& embed, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            return ProjectionDimension;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::LDA;
        }

    private:
        void BetweenScatterCalculation(TVector<float>* result);

    private:
        int TotalDimension;
        int NumClasses;
        int ProjectionDimension;
        float RegParam;
        int Size = 0;
        TVector<IncrementalCloud> ClassesDist;
        TVector<float> ProjectionMatrix;
        TVector<float> EigenValues;
        TVector<float> ProjectionCalculationCache;

    protected:
        friend class TLinearDACalcerVisitor;
    };

    class TLinearDACalcerVisitor final : public IEmbeddingCalcerVisitor {
    public:
        void Update(ui32 classId, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
        void Flush(TEmbeddingFeatureCalcer* featureCalcer);
    private:
        int LastFlush = 0;
    };
};
