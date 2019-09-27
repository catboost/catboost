#pragma once

#include "feature_calcer.h"

#include <catboost/libs/text_processing/dictionary.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NCB {

    class TTextProcessingCollection : public TThrRefBase {
    public:
        TTextProcessingCollection() = default;

        TTextProcessingCollection(
            TVector<TTextFeatureCalcerPtr> calcers,
            TVector<TDictionaryPtr> dictionaries,
            TVector<TVector<ui32>> perFeatureDictionaries,
            TVector<TVector<ui32>> perTokenizedFeatureCalcers,
            TTokenizerPtr tokenizer);

        void CalcFeatures(
            TConstArrayRef<TStringBuf> textFeature,
            ui32 textFeatureIdx,
            TArrayRef<float> result) const {
            CalcFeatures(textFeature, textFeatureIdx, textFeature.size(), result);
        }

        void CalcFeatures(
            TConstArrayRef<TStringBuf> textFeature,
            ui32 textFeatureIdx,
            size_t docCount,
            TArrayRef<float> result) const;

        TStringBuf GetStringIdentifier() {
            return TStringBuf(StringIdentifier.data());
        }

        TTextFeatureCalcerPtr GetCalcer(ui32 calcerId) const {
            return FeatureCalcers[calcerId];
        }

        ui32 NumberOfOutputFeatures(ui32 textFeatureId) const;
        ui32 TotalNumberOfOutputFeatures() const;

        void Save(IOutputStream* s) const;
        void Load(IInputStream* s);

        ui32 GetTokenizedFeatureId(ui32 textFeatureIdx, ui32 dictionaryIdx) const;
        ui32 GetCalcerFeatureOffset(ui32 textFeatureIdx, ui32 dictionaryIdx, ui32 calcerIdx) const;

        bool operator==(const TTextProcessingCollection& rhs);
        bool operator!=(const TTextProcessingCollection& rhs);

    private:
        void SaveHeader(IOutputStream* stream) const;
        void LoadHeader(IInputStream* stream);

        void CalcProcessedFeatureIdx();
        void CheckPerFeatureIdx() const;

        TTokenizerPtr Tokenizer = CreateTokenizer();
        TVector<TDictionaryPtr> Dictionaries;
        TVector<TTextFeatureCalcerPtr> FeatureCalcers;

        TVector<TGuid> DictionaryId;
        TVector<TGuid> FeatureCalcerId;

        TVector<TVector<ui32>> PerFeatureDictionaries;
        TVector<TVector<ui32>> PerTokenizedFeatureCalcers;

        THashMap<std::pair<ui32, ui32>, ui32> TokenizedFeatureId;
        THashMap<std::tuple<ui32, ui32, ui32>, ui32> ProcessedFeatureId;

        static constexpr std::array<char, 16> StringIdentifier = {"TTextCollection"};
        static constexpr size_t IdentifierSize = 16;
        static constexpr ui32 SerializationAlignment = 16;
    };

} // ncb
