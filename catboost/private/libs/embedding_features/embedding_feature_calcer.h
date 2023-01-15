#pragma once

#include <catboost/private/libs/text_features/feature_calcer.h>
#include <catboost/private/libs/embeddings/embedding_dataset.h>


namespace NCB {

    class TEmbeddingFeatureCalcer : public IFeatureCalcer {
    public:
        TEmbeddingFeatureCalcer(ui32 baseFeatureCount, const TGuid& calcerId)
            : ActiveFeatureIndices(baseFeatureCount)
            , Guid(calcerId)
        {
            Iota(ActiveFeatureIndices.begin(), ActiveFeatureIndices.end(), 0);
        }

        virtual void Compute(const TEmbeddingsArray& vector, TOutputFloatIterator outputFeaturesIterator) const = 0;

        //TODO: oganes
        //void Save(IOutputStream* stream) const;
        //void Load(IInputStream* stream);

        TGuid Id() const override {
            return Guid;
        }

        void SetId(const TGuid& guid) {
            Guid = guid;
        }

        void TrimFeatures(TConstArrayRef<ui32> featureIndices) override;
        TConstArrayRef<ui32> GetActiveFeatureIndices() const;

    protected:
        class TFeatureCalcerFbs;

        template <class F>
        void ForEachActiveFeature(F&& func) const {
            for (ui32 featureId: GetActiveFeatureIndices()) {
                func(featureId);
            }
        }

    private:
        TVector<ui32> ActiveFeatureIndices;
        TGuid Guid = CreateGuid();
    };

    class IEmbeddingCalcerVisitor : public TThrRefBase {
    public:
        virtual void Update(ui32 classId, const TEmbeddingsArray& vector, TEmbeddingFeatureCalcer* featureCalcer) = 0;
    };

};

