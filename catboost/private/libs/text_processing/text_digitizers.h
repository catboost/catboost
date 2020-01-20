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
        TTextDigitizers() = default;

        void AddDigitizer(ui32 srcTextIdx, ui32 dstTextIdx, TTokenizerPtr tokenizer, TDictionaryPtr dictionary) {
            CB_ENSURE(
                !Tokenizers.contains(dstTextIdx),
                "Attempt to add rewrite tokenizer for dstTextIdx=" << dstTextIdx
            );
            CB_ENSURE(
                !Dictionaries.contains(dstTextIdx),
                "Attempt to add rewrite dictionary for dstTextIdx=" << dstTextIdx
            );
            SourceToDestinationIndexes[srcTextIdx].insert(dstTextIdx);
            IdToTokenizer[tokenizer->Id()] = tokenizer;
            Tokenizers[dstTextIdx] = std::move(tokenizer);
            IdToDictionary[dictionary->Id()] = dictionary;
            Dictionaries[dstTextIdx] = std::move(dictionary);
        }

        bool HasDigitizer(ui32 dstTextIdx) {
            CB_ENSURE_INTERNAL(
                Tokenizers.contains(dstTextIdx) == Dictionaries.contains(dstTextIdx),
                "Each dictionary should have own tokenizer and vice versa."
            );
            return Tokenizers.contains(dstTextIdx);
        }

        TTokenizerPtr GetTokenizer(ui32 dstTextIdx) const {
            return Tokenizers.at(dstTextIdx);
        }

        TDictionaryPtr GetDictionary(ui32 dstTextIdx) const {
            return Dictionaries.at(dstTextIdx);
        }

        ui32 GetSourceTextsCount() const {
            return SourceToDestinationIndexes.size();
        }

        ui32 GetDigitizedTextsCount() const {
            CB_ENSURE_INTERNAL(
                Tokenizers.size() == Dictionaries.size(),
                "Tokenizers and Dictionaries maps should have the same size."
            );
            return Tokenizers.size();
        }

        ui32 GetDigitizedTextsCount(ui32 sourceTextIdx) const {
            return SourceToDestinationIndexes.at(sourceTextIdx).size();
        }

        template <class TSourceTextAccessor, class TDigitizedTextWriter>
        void Apply(
            TSourceTextAccessor&& sourceTextAccessor,
            TDigitizedTextWriter&& digitizedTextWriter,
            NPar::TLocalExecutor* localExecutor
        ) const {
            TVector<std::pair<ui32, ui32>> sourceToDestinationPairs;
            for (const auto& [sourceTextIdx, digitizedSetIndices]: SourceToDestinationIndexes) {
                const auto sourceText = sourceTextAccessor(sourceTextIdx);

                for (ui32 digitizedTextIdx: digitizedSetIndices) {
                    const auto& dictionary = Dictionaries.at(digitizedTextIdx);
                    const auto& tokenizer = Tokenizers.at(digitizedTextIdx);

                    TTextColumnBuilder textColumnBuilder(tokenizer, dictionary, sourceText.Size());
                    sourceText.ForEach(
                        [&](ui32 index, TStringBuf phrase) {
                            textColumnBuilder.AddText(index, phrase);
                        },
                        localExecutor
                    );

                    digitizedTextWriter(digitizedTextIdx, textColumnBuilder.Build());
                }
            }
        }

        TTokenizerPtr GetTokenizer() const {
            // TODO(nikitxskv): It will be fixed in second part of adding tokenizers to catboost..
            return CreateTokenizer();
        }

        TVector<TDictionaryPtr> GetDictionaries() const {
            TVector<TDictionaryPtr> dictionaries;
            dictionaries.resize(Dictionaries.size());

            for (const auto& [dstTextIdx, dictionary]: Dictionaries) {
                dictionaries[dstTextIdx] = dictionary;
            }

            return dictionaries;
        }

    private:
        // Original text feature index -> Tokenized & Dictionarized feature index
        TMap<ui32, TSet<ui32>> SourceToDestinationIndexes;

        THashMap<TGuid, TTokenizerPtr> IdToTokenizer;
        // Tokenized & Dictionarized feature index -> Tokenizer
        TMap<ui32, TTokenizerPtr> Tokenizers;

        THashMap<TGuid, TDictionaryPtr> IdToDictionary;
        // Tokenized & Dictionarized feature index -> Dictionary
        TMap<ui32, TDictionaryPtr> Dictionaries;
    };

}
