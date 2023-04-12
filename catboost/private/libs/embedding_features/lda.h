#pragma once

#include "embedding_feature_calcer.h"

namespace NCB {

    float CalculateGaussianLikehood(const TEmbeddingsArray& embed,
                                                  const TVector<float>& mean,
                                                  const TVector<float>& scatter);

    void InverseMatrix(TVector<float>* matrix, int dim);

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
            int totalDimension = 2,
            bool isClassification = true,
            int numClasses = 2,
            int projectionDimension = 1,
            float regularization = 0.01,
            bool computeProb = false,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(projectionDimension, calcerId)
            , TotalDimension(totalDimension)
            , IsClassification(isClassification)
            , NumClasses(numClasses)
            , ProjectionDimension(projectionDimension)
            , RegParam(regularization)
            , ComputeProbabilities(computeProb)
            , ClassesDist(isClassification ? numClasses : 1, totalDimension)
            , ProjectionMatrix(totalDimension * projectionDimension)
            , BetweenMatrix(totalDimension * totalDimension)
            , EigenValues(TotalDimension)
            , ProjectionCalculationCache(totalDimension * (totalDimension + 2))
        {}

        void Compute(const TEmbeddingsArray& embed, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            if (ComputeProbabilities) {
                return ProjectionDimension + NumClasses;
            }
            return ProjectionDimension;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::LDA;
        }

    protected:
        TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const override;
        void LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer* calcerFbs) override;

        void SaveLargeParameters(IOutputStream*) const override;
        void LoadLargeParameters(IInputStream*) override;

    private:
        void TotalScatterCalculation(TVector<float>* result);

    private:
        int TotalDimension;
        bool IsClassification;
        int NumClasses; // used only if IsClassification == true
        int ProjectionDimension;
        float RegParam;
        bool ComputeProbabilities; // used only if IsClassification == true
        int Size = 0;
        TVector<IncrementalCloud> ClassesDist; // if IsClassification == false (i.e. regression) contains only one element
        TVector<float> ProjectionMatrix;
        TVector<float> BetweenMatrix;
        TVector<float> EigenValues;
        TVector<float> ProjectionCalculationCache;

    protected:
        friend class TLinearDACalcerVisitor;
    };

    class TLinearDACalcerVisitor final : public IEmbeddingCalcerVisitor {
    public:
        void Update(float target, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
        void Flush(TEmbeddingFeatureCalcer* featureCalcer);
    private:
        int LastFlush = 0;
    };
};
