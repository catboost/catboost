#pragma once

#include "types.h"

#include <library/cpp/containers/flat_hash/flat_hash.h>

#include <util/generic/string.h>

namespace NTextProcessing::NDictionary {

    using TInternalTokenId = TTokenId;
    static const TString END_OF_SENTENCE_SYMBOL = "";
    constexpr TInternalTokenId UNKNOWN_INTERNAL_TOKEN_ID = Max<TInternalTokenId>();
    static const TString DICT_FORMAT_KEY = "dictionary_format";
    static const TString DICT_NEW_FORMAT_DESC = "id_count_token";

    template <ui32 N>
    struct TMultiInternalTokenId : public std::array<TInternalTokenId, N> {
    };

    template <ui32 GramOrder, typename TValue>
    using TInternalIdsMap = NFH::TFlatHashMap<TMultiInternalTokenId<GramOrder>, TValue>;

    template <typename TTokenType>
    TInternalTokenId GetInternalWordTokenId(
        const TTokenType& token,
        NFH::TFlatHashMap<TString, TInternalTokenId>* tokenToInternalId
    ) {
        const auto it = tokenToInternalId->find(token);
        if (it != tokenToInternalId->end()) {
            return it->second;
        } else {
            const TInternalTokenId tokenId = tokenToInternalId->size();
            tokenToInternalId->emplace(token, tokenId);
            return tokenId;
        }
    }

    template <typename T>
    class TConstArrayRefChainer {
    public:
        TConstArrayRefChainer(TConstArrayRef<T> firstContainer, TConstArrayRef<T> secondContainer)
            : FirstContainer(firstContainer)
            , SecondContainer(secondContainer)
        {
        }

        size_t Size() const {
            return FirstContainer.size() + SecondContainer.size();
        }

        const T& operator[](ui32 i) const {
            return i < FirstContainer.size() ? FirstContainer[i] : SecondContainer[i - FirstContainer.size()];
        }

    private:
        TConstArrayRef<T> FirstContainer;
        TConstArrayRef<T> SecondContainer;
    };

    template <typename TTokenType>
    TConstArrayRefChainer<TTokenType> AppendEndOfSentenceTokenIfNeed(
        TConstArrayRef<TTokenType> rawTokens,
        EEndOfSentenceTokenPolicy endOfSentenceTokenPolicy,
        TVector<TTokenType>* vectorWithEndOfSentence
    ) {
        vectorWithEndOfSentence->clear();
        if (endOfSentenceTokenPolicy == EEndOfSentenceTokenPolicy::Insert) {
            vectorWithEndOfSentence->emplace_back(END_OF_SENTENCE_SYMBOL);
        }
        return {rawTokens, *vectorWithEndOfSentence};
    }

    inline ui32 GetEndTokenIndex(ui32 tokenCount, ui32 gramOrder, ui32 skipStep) {
        const ui32 lastGramTokenIndex = (gramOrder - 1) * (skipStep + 1);
        return lastGramTokenIndex < tokenCount ? tokenCount - lastGramTokenIndex : 0;
    }

}
