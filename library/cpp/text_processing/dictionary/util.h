#pragma once

#include <util/charset/utf8.h>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/string.h>
#include <util/generic/ylimits.h>
#include <util/generic/vector.h>

namespace NTextProcessing::NDictionary {

    inline ui32 GetMaxDictionarySize(i32 maxDictionarySize) {
        Y_ENSURE(maxDictionarySize > 0 || maxDictionarySize == -1,
            "Incorrect max dictionary size value: " << maxDictionarySize
            << ". It should be greater 0 or equal -1.");
        return maxDictionarySize == -1 ? Max<ui32>() : maxDictionarySize;
    }

    template <typename TTokenType>
    static void GetLetterIndices(const TTokenType& token, TVector<ui32>* letterStartIndices) {
        letterStartIndices->clear();
        const unsigned char* current = reinterpret_cast<const unsigned char*>(token.data());
        const ui32 tokenSize = token.size();
        const unsigned char* last = current + tokenSize;
        ui32 i = 0;
        while (i < tokenSize) {
            letterStartIndices->push_back(i);
            size_t length = 0;
            auto recodeResult = GetUTF8CharLen(length, current, last);
            if (recodeResult != RECODE_OK || length == 0) {
                letterStartIndices->clear();
                return;
            }
            i += length;
            current += length;
        }
        letterStartIndices->push_back(tokenSize);
    }

    template <bool needToAddEndOfWordToken, typename TTokenType, typename TVisitor>
    static void ApplyFuncToLetterNGramsImpl(
        TConstArrayRef<TTokenType> tokens,
        int gramOrder,
        TVisitor& visitor
    ) {
        TVector<ui32> letterStartIndices;
        for (const auto& token : tokens) {
            GetLetterIndices(token, &letterStartIndices);
            if (letterStartIndices.empty()) {
                continue;
            }

            const int lettersCount = letterStartIndices.size() - 1; // Last element of this vector is token.size()
            const int gramCount = lettersCount - gramOrder + 1;

            // Add start of word token (First gramOrder - 1 symbols)
            if (gramOrder <= lettersCount + 1) {
                visitor({token.data(), token.data() + letterStartIndices[gramOrder - 1]});
            }

            for (int i = 0; i < gramCount; ++i) {
                visitor({
                    token.data() + letterStartIndices[i],
                    token.data() + letterStartIndices[i + gramOrder]
                });
            }

            if constexpr (needToAddEndOfWordToken) {
                // Add end of word token (Last gramOrder - 1 symbols + " ")
                if (gramOrder <= lettersCount + 2) {
                    const int lastGramBegin = Max<int>(0, lettersCount + 1 - gramOrder);
                    visitor(TString(
                        token.data() + letterStartIndices[lastGramBegin],
                        token.data() + token.size()
                    ) + " ");
                }
            }
        }
    }

    template <typename TTokenType, typename TVisitor>
    void ApplyFuncToLetterNGrams(
        TConstArrayRef<TTokenType> tokens,
        ui32 gramOrder,
        bool needToAddEndOfWordToken,
        TVisitor& visitor
    ) {
        if (needToAddEndOfWordToken) {
            ApplyFuncToLetterNGramsImpl<true>(tokens, gramOrder, visitor);
        } else {
            ApplyFuncToLetterNGramsImpl<false>(tokens, gramOrder, visitor);
        }
    }

}
