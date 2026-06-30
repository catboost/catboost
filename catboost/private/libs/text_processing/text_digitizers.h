#pragma once

#include "dictionary.h"
#include "text_column_builder.h"

#include <catboost/libs/helpers/guid.h>

#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/str_stl.h>

namespace NCB {

    struct TDigitizerId {
        TGuid TokenizerId;
        TGuid DictionaryId;

        bool operator==(const TDigitizerId& other) const {
            return std::tie(TokenizerId, DictionaryId) == std::tie(other.TokenizerId, other.DictionaryId);
        }
    };

    struct TDigitizer {
        TTokenizerPtr Tokenizer;
        TDictionaryPtr Dictionary;

        TDigitizerId Id() const {
            return {Tokenizer->Id(), Dictionary->Id()};
        }

        bool operator==(const TDigitizer& other) const {
            return Id() == other.Id();
        }
    };

    class TTextDigitizers {
    public:
        TTextDigitizers() = default;

        void AddDigitizer(ui32 srcTextIdx, ui32 dstTextIdx, TDigitizer digitizer) {
            CB_ENSURE(
                !Digitizers.contains(dstTextIdx),
                "Attempt to add rewrite digitizer for dstTextIdx=" << dstTextIdx
            );
            SourceToDestinationIndexes[srcTextIdx].insert(dstTextIdx);
            Digitizers[dstTextIdx] = std::move(digitizer);
        }

        bool HasDigitizer(ui32 dstTextIdx) const {
            return Digitizers.contains(dstTextIdx);
        }

        TDigitizer GetDigitizer(ui32 dstTextIdx) const {
            return Digitizers.at(dstTextIdx);
        }

        ui32 GetSourceTextsCount() const {
            if (SourceToDestinationIndexes.empty()) {
                return 0;
            }
            return *SourceToDestinationIndexes.begin()->second.begin(); // index of 1st tokenized text feature == number of text features
        }

        ui32 GetDigitizedTextsCount() const {
            return Digitizers.size();
        }

        ui32 GetDigitizedTextsCount(ui32 sourceTextIdx) const {
            return SourceToDestinationIndexes.at(sourceTextIdx).size();
        }

        template <class TSourceTextAccessor, class TDigitizedTextWriter>
        void Apply(
            TSourceTextAccessor&& sourceTextAccessor,
            TDigitizedTextWriter&& digitizedTextWriter,
            NPar::ILocalExecutor* localExecutor
        ) const {
            TVector<std::pair<ui32, ui32>> sourceToDestinationPairs;
            for (const auto& [sourceTextIdx, digitizedSetIndices]: SourceToDestinationIndexes) {
                const auto sourceText = sourceTextAccessor(sourceTextIdx);

                for (ui32 digitizedTextIdx: digitizedSetIndices) {
                    const auto& dictionary = Digitizers.at(digitizedTextIdx).Dictionary;
                    const auto& tokenizer = Digitizers.at(digitizedTextIdx).Tokenizer;

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

        TVector<TDigitizer> GetDigitizers() const {
            TVector<TDigitizer> digitizers;
            for (const auto& [dstTextIdx, digitizer]: Digitizers) {
                Y_UNUSED(dstTextIdx);
                digitizers.push_back(digitizer);
            }

            return digitizers;
        }

    private:
        // Original text feature index -> Tokenized & Dictionarized feature index
        TMap<ui32, TSet<ui32>> SourceToDestinationIndexes;

        // Tokenized & Dictionarized feature index -> Digitizer
        TMap<ui32, TDigitizer> Digitizers;
    };

}

template <>
struct THash<NCB::TDigitizer> {
    size_t operator()(const NCB::TDigitizer& digitizer) const noexcept {
        return THash<std::pair<NCB::TGuid, NCB::TGuid>>()(std::make_pair(
            digitizer.Tokenizer->Id(),
            digitizer.Dictionary->Id()
        ));
    }
};
