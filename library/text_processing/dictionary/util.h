#pragma once

#include <util/charset/utf8.h>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NTextProcessing::NDictionary {

    inline ui32 GetMaxDictionarySize(i32 maxDictionarySize) {
        Y_ENSURE(maxDictionarySize > 0 || maxDictionarySize == -1,
            "Incorrect max dictionary size value: " << maxDictionarySize
            << ". It should be greater 0 or equal -1.");
        return maxDictionarySize == -1 ? Max<ui32>() : maxDictionarySize;
    }

    class LetterGenerator {
    public:
        LetterGenerator(TStringBuf token)
            : Token(token)
            , Current(reinterpret_cast<const unsigned char*>(token.data()))
            , Last(Current + token.size())
        {
        }

        TStringBuf GetNextLetter() {
            size_t length;
            GetUTF8CharLen(length, Current, Last);
            length = Min(length, static_cast<size_t>(Last - Current));
            TStringBuf result(reinterpret_cast<const char*>(Current), length);
            Current += length;
            return result;
        }

        bool HasNextLetter() const {
            return Current != Last;
        }

    private:
        TStringBuf Token;
        const unsigned char* Current;
        const unsigned char* Last;
    };


    template <typename TVisitor>
    static void MaybeUpdateLetterNGram(
        TStringBuf letter,
        ui32 gramOrder,
        ui32 skipStep,
        TDeque<TStringBuf>* letters,
        TVisitor& visitor
    ) {
        letters->push_back(letter);
        if ((gramOrder - 1) * (skipStep + 1) == letters->size() - 1) {
            TString letterNGram((*letters)[0]);
            for (ui32 gramIndex = 1; gramIndex < gramOrder; ++gramIndex) {
                letterNGram += (*letters)[(gramIndex) * (skipStep + 1)];
            }
            visitor(letterNGram);
            letters->pop_front();
        }
    }

    template <typename TTokenType, typename TVisitor>
    static void ApplyFuncToLetterNGramsAddEOW(
        TConstArrayRef<TTokenType> tokens,
        ui32 gramOrder,
        ui32 skipStep,
        TVisitor& visitor
    ) {
        TDeque<TStringBuf> letters;
        static const TString spaceSymbol = " ";
        MaybeUpdateLetterNGram(spaceSymbol, gramOrder, skipStep, &letters, visitor);
        for (const auto& token : tokens) {
            LetterGenerator lettersGenerator(token);
            while (lettersGenerator.HasNextLetter()) {
                MaybeUpdateLetterNGram(lettersGenerator.GetNextLetter(), gramOrder, skipStep, &letters, visitor);
            }
            MaybeUpdateLetterNGram(spaceSymbol, gramOrder, skipStep, &letters, visitor);
        }
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
            size_t length;
            GetUTF8CharLen(length, current, last);
            i += length;
            current += length;
        }
        letterStartIndices->push_back(tokenSize);
    }

    template <typename TTokenType, typename TVisitor>
    static void ApplyFuncToLetterNGramsSkipEOW(
        TConstArrayRef<TTokenType> tokens,
        ui32 gramOrder,
        ui32 skipStep,
        TVisitor& visitor
    ) {
        TVector<ui32> letterStartIndices;
        TString letterNGram;
        for (const auto& token : tokens) {
            GetLetterIndices(token, &letterStartIndices);
            const ui32 lettersCount = letterStartIndices.size() - 1; // Last element of this vector is token.size()
            const ui32 windowSize = (gramOrder - 1) * (skipStep + 1) + 1;
            if (windowSize > lettersCount) {
                continue;
            }
            const ui32 gramCount = lettersCount - windowSize + 1;
            for (ui32 i = 0; i < gramCount; ++i) {
                letterNGram.clear();
                letterNGram.insert(
                    letterNGram.end(),
                    token.data() + letterStartIndices[i],
                    token.data() + letterStartIndices[i + 1]
                );
                for (ui32 gramIndex = 1; gramIndex < gramOrder; ++gramIndex) {
                    const ui32 letterIndex = i + gramIndex * (skipStep + 1);
                    letterNGram.insert(
                        letterNGram.end(),
                        token.data() + letterStartIndices[letterIndex],
                        token.data() + letterStartIndices[letterIndex + 1]
                    );
                }
                visitor(letterNGram);
            }
        }
    }

    template <typename TTokenType, typename TVisitor>
    void ApplyFuncToLetterNGrams(
        TConstArrayRef<TTokenType> tokens,
        ui32 gramOrder,
        ui32 skipStep,
        bool needToAddEndOfWordToken,
        TVisitor& visitor
    ) {
        if (needToAddEndOfWordToken) {
            ApplyFuncToLetterNGramsAddEOW(tokens, gramOrder, skipStep, visitor);
        } else {
            ApplyFuncToLetterNGramsSkipEOW(tokens, gramOrder, skipStep, visitor);
        }
    }

}
