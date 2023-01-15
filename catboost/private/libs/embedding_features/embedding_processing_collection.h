#pragma once

#include "embedding_feature_calcer.h"

#include <catboost/private/libs/text_features/text_processing_collection.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NCB {

    class TEmbeddingProcessingCollection : public TThrRefBase {
    public:
        TEmbeddingProcessingCollection() = default;

        void CalcFeatures(
            TConstArrayRef<TEmbeddingsArray> embeddingFeature,
            ui32 embeddingFeatureIdx,
            TArrayRef<float> result) const;

        template <class TEmbeddingFeatureAccessor>
        void CalcFeatures(
            TEmbeddingFeatureAccessor featureAccessor,
            TConstArrayRef<ui32> embeddingFeatureIds,
            ui32 docCount,
            TArrayRef<float> result
        ) const {
            const ui32 totalNumberOfFeatures = TotalNumberOfOutputFeatures() * docCount;
            CB_ENSURE(
                result.size() >= totalNumberOfFeatures,
                "Proposed result buffer has size (" << result.size()
                    << ") less than embedding processing produce (" << totalNumberOfFeatures << ')'
            );

            TVector<TEmbeddingsArray> embeddings;
            embeddings.yresize(docCount);

            float* estimatedFeatureBegin = &result[0];
            for (ui32 embeddingFeatureId: embeddingFeatureIds) {
                auto estimatedFeatureEnd = estimatedFeatureBegin + NumberOfOutputFeatures(embeddingFeatureId) * docCount;

                for (size_t docId : xrange(docCount)) {
                    static_assert(std::is_same<decltype(featureAccessor(embeddingFeatureId, docId)), TConstArrayRef<float>>::value);
                    embeddings[docId] = TMaybeOwningArrayHolder<const float>::CreateNonOwning(featureAccessor(embeddingFeatureId, docId));
                }

                CalcFeatures(
                    MakeConstArrayRef(embeddings),
                    embeddingFeatureId,
                    TArrayRef<float>(
                        estimatedFeatureBegin,
                        estimatedFeatureEnd
                    )
                );
                estimatedFeatureBegin = estimatedFeatureEnd;
            }
        }


        TEmbeddingProcessingCollection(
            TVector<TEmbeddingFeatureCalcerPtr> calcers,
            TVector<TVector<ui32>> perEmbeddingFeatureCalcers
        );

        ui32 GetAbsoluteCalcerOffset(const TGuid& calcerGuid) const;
        ui32 GetRelativeCalcerOffset(ui32 FeatureIdx, const TGuid& calcerGuid) const;

        ui32 GetEmbeddingFeatureCount() const;
        ui32 GetTokenizedFeatureCount() const;

        TEmbeddingFeatureCalcerPtr GetCalcer(ui32 calcerId) const {
            return FeatureCalcers[calcerId];
        }

        TEmbeddingFeatureCalcerPtr GetCalcer(const TGuid& calcerGuid) const {
            return FeatureCalcers[CalcerGuidToFlatIdx.at(calcerGuid)];
        }

        ui32 NumberOfOutputFeatures(ui32 featureId) const;
        ui32 TotalNumberOfOutputFeatures() const;

        TVector<TEvaluatedFeature> GetProducedFeatures() const;

        void Save(IOutputStream* s) const;
        void Load(IInputStream* stream);
        void LoadNonOwning(TMemoryInput* in);
        void DefaultInit(TCountingInput s);

        bool operator==(const TEmbeddingProcessingCollection& rhs);
        bool operator!=(const TEmbeddingProcessingCollection& rhs);

        static TString GetStringIdentifier() {
            return TString(StringIdentifier.data(), StringIdentifier.size());
        }

        bool Empty() {
            return FeatureCalcers.empty();
        }

    private:
        ui32 GetEmbeddingFeatureId(ui32 textFeatureIdx) const;

        void SaveHeader(IOutputStream* stream) const;
        void LoadHeader(IInputStream* stream);

        void CalcRuntimeData();
        void CheckPerFeatureIdx() const;

        ui32 GetAbsoluteCalcerOffset(ui32 calcerGuid) const;
        ui32 GetRelativeCalcerOffset(ui32 FeatureIdx, ui32 calcerGuid) const;

    private:
        TVector<TEmbeddingFeatureCalcerPtr> FeatureCalcers;
        TVector<TGuid> FeatureCalcerId;
        TVector<TVector<ui32>> PerEmbeddingFeatureCalcers;

        THashMap<TGuid, ui32> CalcerGuidToFlatIdx;
        THashMap<ui32, ui32> FeatureCalcerOffset;

        static constexpr std::array<char, 16> StringIdentifier = {"embed_process_1"};
        static constexpr size_t IdentifierSize = 16;
        static constexpr ui32 SerializationAlignment = 16;
    };
}
