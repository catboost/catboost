#pragma once

#include <catboost/private/libs/text_features/feature_calcer.h>
#include <catboost/private/libs/embeddings/embedding_dataset.h>

#include <catboost/private/libs/embedding_features/flatbuffers/embedding_feature_calcers.fbs.h>


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

        void Save(IOutputStream* stream) const final;
        void Load(IInputStream* stream) final;

        TGuid Id() const override {
            return Guid;
        }

        void SetId(const TGuid& guid) {
            Guid = guid;
        }

        void TrimFeatures(TConstArrayRef<ui32> featureIndices) override;
        TConstArrayRef<ui32> GetActiveFeatureIndices() const;

    protected:
        class TEmbeddingCalcerFbs {
        public:
            using TCalcerFbsImpl = flatbuffers::Offset<void>;
            using ECalcerFbsType = NCatBoostFbs::NEmbeddings::TAnyEmbeddingCalcer;

            TEmbeddingCalcerFbs(
                ECalcerFbsType calcerFbsType,
                TCalcerFbsImpl featureCalcerFbs
            )
                : CalcerType(calcerFbsType)
                , CalcerFlatBuffer(featureCalcerFbs)
            {}

            ECalcerFbsType GetCalcerType() const {
                return CalcerType;
            }

            const TCalcerFbsImpl& GetCalcerFlatBuffer() const {
                return CalcerFlatBuffer;
            }

        private:
            ECalcerFbsType CalcerType;
            TCalcerFbsImpl CalcerFlatBuffer;
        };

        template <class F>
        void ForEachActiveFeature(F&& func) const {
            for (ui32 featureId: GetActiveFeatureIndices()) {
                func(featureId);
            }
        }

        NCatBoostFbs::TGuid GetFbsGuid() const {
            return CreateFbsGuid(Guid);
        }

        virtual TEmbeddingCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const;
        virtual void SaveLargeParameters(IOutputStream*) const;

        virtual void LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer*);
        virtual void LoadLargeParameters(IInputStream*);

        flatbuffers::Offset<flatbuffers::Vector<uint32_t>> ActiveFeatureIndicesToFB(flatbuffers::FlatBufferBuilder& builder) const;

    private:
        TVector<ui32> ActiveFeatureIndices;
        TGuid Guid = CreateGuid();
    };

    class IEmbeddingCalcerVisitor : public TThrRefBase {
    public:
        virtual void Update(float target, const TEmbeddingsArray& vector, TEmbeddingFeatureCalcer* featureCalcer) = 0;
    };

    using TEmbeddingFeatureCalcerPtr = TIntrusivePtr<TEmbeddingFeatureCalcer>;
    using TEmbeddingFeatureCalcerFactory = NObjectFactory::TParametrizedObjectFactory<TEmbeddingFeatureCalcer,
                                                                                      EFeatureCalcerType>;

    class TEmbeddingCalcerSerializer {
    public:
        static void Save(IOutputStream* stream, const TEmbeddingFeatureCalcer& calcer);
        static TEmbeddingFeatureCalcerPtr Load(IInputStream* stream);

    private:
        static constexpr std::array<char, 16> CalcerMagic = {"EmbedCalcerV1"};
        static constexpr ui32 MagicSize = CalcerMagic.size();
        static constexpr ui32 Alignment = 16;
    };
};

