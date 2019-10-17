#pragma once

#include "dictionary.h"
#include "text_column_builder.h"

#include <catboost/libs/helpers/guid.h>

#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>

namespace NCB {

    class TTextDigitizers {
    public:
        TTextDigitizers() {
            Tokenizer = CreateTokenizer();
        }

        void AddDictionary(ui32 srcTextIdx, ui32 dstTextIdx, TDictionaryPtr dictionary) {
            CB_ENSURE(
                !Dictionaries.contains(dstTextIdx),
                "Attempt to add rewrite dictionary for dstTextIdx=" << dstTextIdx
            );
            SourceToDestinationIndexes[srcTextIdx].insert(dstTextIdx);
            IdToDictionary[dictionary->Id()] = dictionary;
            Dictionaries[dstTextIdx] = std::move(dictionary);
        }

        bool HasDictionary(ui32 dstTextIdx) {
            return Dictionaries.contains(dstTextIdx);
        }

        TDictionaryPtr GetDictionary(const TGuid& guid) const {
            return IdToDictionary.at(guid);
        }
        TDictionaryPtr GetDictionary(ui32 dstTextIdx) const {
            return Dictionaries.at(dstTextIdx);
        }

        TTokenizerPtr GetTokenizer() const {
            return Tokenizer;
        }

        ui32 GetSourceTextsCount() const {
            return SourceToDestinationIndexes.size();
        }

        ui32 GetDigitizedTextsCount() const {
            return Dictionaries.size();
        }

        ui32 GetDigitizedTextsCount(ui32 sourceTextIdx) const {
            return SourceToDestinationIndexes.at(sourceTextIdx).size();
        }

        template <class TSourceTextAccessor, class TDigitizedTextWriter>
        void Apply(TSourceTextAccessor&& sourceTextAccessor, TDigitizedTextWriter&& digitizedTextWriter) const {
            for (const auto& [sourceTextIdx, digitizedSetIndices]: SourceToDestinationIndexes) {
                const auto sourceText = sourceTextAccessor(sourceTextIdx);

                for (ui32 digitizedTextIdx: digitizedSetIndices) {
                    const auto& dictionary = Dictionaries.at(digitizedTextIdx);

                    TTextColumnBuilder textColumnBuilder(Tokenizer, dictionary, sourceText.Size());
                    sourceText.ForEach(
                        [&](ui32 index, TStringBuf phrase) {
                            textColumnBuilder.AddText(index, phrase);
                        }
                    );

                    digitizedTextWriter(digitizedTextIdx, textColumnBuilder.Build());
                }
            }
        }

        TVector<TDictionaryPtr> GetDictionaries() {
            TVector<TDictionaryPtr> dictionaries;
            dictionaries.resize(Dictionaries.size());

            for (const auto& [dstTextIdx, dictionary]: Dictionaries) {
                dictionaries[dstTextIdx] = dictionary;
            }

            return dictionaries;
        }

    private:
        THashMap<TGuid, TDictionaryPtr> IdToDictionary;
        TMap<ui32, TSet<ui32>> SourceToDestinationIndexes;
        TMap<ui32, TDictionaryPtr> Dictionaries;
        TTokenizerPtr Tokenizer;
    };

}
