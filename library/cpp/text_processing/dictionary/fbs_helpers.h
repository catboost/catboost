#pragma once

#include "options.h"

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <library/cpp/text_processing/dictionary/idl/dictionary_meta_info.fbs.h>

#include <util/generic/string.h>

namespace NTextProcessing::NDictionary {

    inline NTextProcessingFbs::ETokenLevelType ToFbs(ETokenLevelType tokenLevelType) {
        switch (tokenLevelType) {
            case ETokenLevelType::Word:
                return NTextProcessingFbs::ETokenLevelType_Word;
            case ETokenLevelType::Letter:
                return NTextProcessingFbs::ETokenLevelType_Letter;
            default:
                Y_UNREACHABLE();
        }
    }

    inline NTextProcessingFbs::EEndOfWordTokenPolicy ToFbs(EEndOfWordTokenPolicy endOfWordTokenPolicy) {
        switch (endOfWordTokenPolicy) {
            case EEndOfWordTokenPolicy::Insert:
                return NTextProcessingFbs::EEndOfWordTokenPolicy_Insert;
            case EEndOfWordTokenPolicy::Skip:
                return NTextProcessingFbs::EEndOfWordTokenPolicy_Skip;
            default:
                Y_UNREACHABLE();
        }
    }

    inline NTextProcessingFbs::EEndOfSentenceTokenPolicy ToFbs(EEndOfSentenceTokenPolicy endOfSentenceTokenPolicy) {
        switch (endOfSentenceTokenPolicy) {
            case EEndOfSentenceTokenPolicy::Insert:
                return NTextProcessingFbs::EEndOfSentenceTokenPolicy_Insert;
            case EEndOfSentenceTokenPolicy::Skip:
                return NTextProcessingFbs::EEndOfSentenceTokenPolicy_Skip;
            default:
                Y_UNREACHABLE();
        }
    }

    inline ETokenLevelType FromFbs(NTextProcessingFbs::ETokenLevelType tokenLevelType) {
        switch (tokenLevelType) {
            case NTextProcessingFbs::ETokenLevelType_Word:
                return ETokenLevelType::Word;
            case NTextProcessingFbs::ETokenLevelType_Letter:
                return ETokenLevelType::Letter;
            default:
                Y_UNREACHABLE();
        }
    }

    inline EEndOfWordTokenPolicy FromFbs(NTextProcessingFbs::EEndOfWordTokenPolicy endOfWordTokenPolicy) {
        switch (endOfWordTokenPolicy) {
            case NTextProcessingFbs::EEndOfWordTokenPolicy_Insert:
                return EEndOfWordTokenPolicy::Insert;
            case NTextProcessingFbs::EEndOfWordTokenPolicy_Skip:
                return EEndOfWordTokenPolicy::Skip;
            default:
                Y_UNREACHABLE();
        }
    }

    inline EEndOfSentenceTokenPolicy FromFbs(NTextProcessingFbs::EEndOfSentenceTokenPolicy endOfSentenceTokenPolicy) {
        switch (endOfSentenceTokenPolicy) {
            case NTextProcessingFbs::EEndOfSentenceTokenPolicy_Insert:
                return EEndOfSentenceTokenPolicy::Insert;
            case NTextProcessingFbs::EEndOfSentenceTokenPolicy_Skip:
                return EEndOfSentenceTokenPolicy::Skip;
            default:
                Y_UNREACHABLE();
        }
    }

    inline void BuildDictionaryMetaInfo(
        ui32 size,
        const TDictionaryOptions& dictionaryOptions,
        TVector<ui8>* buffer
    ) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Align(16);
        NTextProcessingFbs::TDictionaryOptionsBuilder optionsBuilder(builder);
        optionsBuilder.add_TokenLevelType(ToFbs(dictionaryOptions.TokenLevelType));
        optionsBuilder.add_GramOrder(dictionaryOptions.GramOrder);
        optionsBuilder.add_SkipStep(dictionaryOptions.SkipStep);
        optionsBuilder.add_StartTokenId(dictionaryOptions.StartTokenId);
        optionsBuilder.add_EndOfWordTokenPolicy(ToFbs(dictionaryOptions.EndOfWordTokenPolicy));
        optionsBuilder.add_EndOfSentenceTokenPolicy(ToFbs(dictionaryOptions.EndOfSentenceTokenPolicy));
        const auto dictionaryMetaInfoOffset = CreateTDictionaryMetaInfo(
            builder,
            size,
            optionsBuilder.Finish(),
            size + dictionaryOptions.StartTokenId,
            size + dictionaryOptions.StartTokenId + 1
        );
        builder.Finish(dictionaryMetaInfoOffset);
        buffer->resize(builder.GetSize());
        std::memcpy(buffer->data(), builder.GetBufferPointer(), builder.GetSize());
    }

}

