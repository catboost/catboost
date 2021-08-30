#pragma once

#include "feature_calcer.h"

#include <catboost/private/libs/text_processing/dictionary.h>
#include <catboost/private/libs/text_processing/text_digitizers.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NCB {
    struct TEvaluatedFeature {
    public:
        TEvaluatedFeature(
            ui32 featureId,
            TGuid calcerId,
            ui32 localId)
            : FeatureId(featureId)
            , CalcerId(std::move(calcerId))
            , LocalId(localId)
        {}

    public:
        ui32 FeatureId;
        TGuid CalcerId;
        ui32 LocalId;
    };

    class TTextProcessingCollection : public TThrRefBase {
    public:
        TTextProcessingCollection() = default;

        TTextProcessingCollection(
            TVector<TDigitizer> digitizers,
            TVector<TTextFeatureCalcerPtr> calcers,
            TVector<TVector<ui32>> perFeatureDigitizers,
            TVector<TVector<ui32>> perTokenizedFeatureCalcers
        );

        void CalcFeatures(
            TConstArrayRef<TStringBuf> textFeature,
            ui32 textFeatureIdx,
            TArrayRef<float> result) const {
            CalcFeatures(textFeature, textFeatureIdx, textFeature.size(), result);
        }

        template <class TTextFeatureAccessor>
        void CalcFeatures(
            TTextFeatureAccessor featureAccessor,
            TConstArrayRef<ui32> textFeatureIds,
            ui32 docCount,
            TArrayRef<float> result
        ) const {
            const ui32 totalNumberOfFeatures = TotalNumberOfOutputFeatures() * docCount;
            CB_ENSURE(
                result.size() >= totalNumberOfFeatures,
                "Proposed result buffer has size (" << result.size()
                    << ") less than text processing produce (" << totalNumberOfFeatures << ')'
            );

            TVector<TStringBuf> texts;
            texts.yresize(docCount);

            float* estimatedFeatureBegin = &result[0];
            for (ui32 textFeatureId: textFeatureIds) {
                auto estimatedFeatureEnd = estimatedFeatureBegin + NumberOfOutputFeatures(textFeatureId) * docCount;

                for (size_t docId : xrange(docCount)) {
                    static_assert(std::is_same<decltype(featureAccessor(textFeatureId, docId)), TStringBuf>::value);
                    texts[docId] = featureAccessor(textFeatureId, docId);
                }

                CalcFeatures(
                    MakeConstArrayRef(texts),
                    textFeatureId,
                    TArrayRef<float>(
                        estimatedFeatureBegin,
                        estimatedFeatureEnd
                    )
                );
                estimatedFeatureBegin = estimatedFeatureEnd;
            }
        }

        void CalcFeatures(
            TConstArrayRef<TStringBuf> textFeature,
            ui32 textFeatureIdx,
            size_t docCount,
            TArrayRef<float> result) const;

        ui32 GetAbsoluteCalcerOffset(const TGuid& calcerGuid) const;
        ui32 GetRelativeCalcerOffset(ui32 textFeatureIdx, const TGuid& calcerGuid) const;

        ui32 GetTextFeatureCount() const;
        ui32 GetTokenizedFeatureCount() const;

        static TString GetStringIdentifier() {
            return TString(StringIdentifier.data(), StringIdentifier.size());
        }

        TTextFeatureCalcerPtr GetCalcer(ui32 calcerId) const {
            return FeatureCalcers[calcerId];
        }

        TTextFeatureCalcerPtr GetCalcer(const TGuid& calcerGuid) const {
            return FeatureCalcers[CalcerGuidToFlatIdx.at(calcerGuid)];
        }

        ui32 NumberOfOutputFeatures(ui32 textFeatureId) const;
        ui32 TotalNumberOfOutputFeatures() const;

        TVector<TEvaluatedFeature> GetProducedFeatures() const;

        void Save(IOutputStream* s) const;
        void Load(IInputStream* stream);
        void LoadNonOwning(TMemoryInput* in);
        void DefaultInit(TCountingInput s);

        bool operator==(const TTextProcessingCollection& rhs);
        bool operator!=(const TTextProcessingCollection& rhs);

        bool Empty() {
            return FeatureCalcers.empty();
        }

    private:
        ui32 GetFirstTextFeatureCalcer(ui32 textFeatureIdx) const;
        ui32 GetTokenizedFeatureId(ui32 textFeatureIdx, ui32 digitizerIdx) const;

        ui32 GetAbsoluteCalcerOffset(ui32 calcerIdx) const;
        ui32 GetRelativeCalcerOffset(ui32 textFeatureIdx, ui32 calcerIdx) const;

        void SaveHeader(IOutputStream* stream) const;
        void LoadHeader(IInputStream* stream);
        THashMap<TGuid, ui32> CreateComponentGuidsMapping() const;
        void CheckForMissingParts() const;

        void CalcRuntimeData();
        void CheckPerFeatureIdx() const;

        TVector<TDigitizer> Digitizers;
        TVector<TTextFeatureCalcerPtr> FeatureCalcers;

        TVector<TGuid> TokenizerId;
        TVector<TGuid> DictionaryId;
        TVector<TGuid> FeatureCalcerId;
        THashMap<TGuid, ui32> CalcerGuidToFlatIdx;

        TVector<TVector<ui32>> PerFeatureDigitizers;
        TVector<TVector<ui32>> PerTokenizedFeatureCalcers;

        THashMap<std::pair<ui32, ui32>, ui32> TokenizedFeatureId;
        THashMap<ui32, ui32> FeatureCalcerOffset;

        static constexpr std::array<char, 16> StringIdentifier = {"text_process_v2"};
        static constexpr size_t IdentifierSize = 16;
        static constexpr ui32 SerializationAlignment = 16;
    };

} // ncb
