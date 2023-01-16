#include "dictionary_builder.h"

#include "frequency_based_dictionary_impl.h"
#include "util.h"

#include <util/charset/utf8.h>
#include <util/generic/deque.h>

namespace NTextProcessing::NDictionary {
    class IDictionaryBuilderImpl {
    public:
        IDictionaryBuilderImpl(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions
        )
            : DictionaryBuilderOptions(dictionaryBuilderOptions)
            , DictionaryOptions(dictionaryOptions)
        {
        }

        void Add(TStringBuf token, ui64 weight) {
            TVector<TStringBuf> tokens = {token};
            Add(tokens, weight);
        }
        virtual void Add(TConstArrayRef<TString> tokens, ui64 weight) = 0;
        virtual void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) = 0;
        virtual TIntrusivePtr<TDictionary> FinishBuilding() = 0;

        virtual ~IDictionaryBuilderImpl() = default;
    protected:
        TDictionaryBuilderOptions DictionaryBuilderOptions;
        TDictionaryOptions DictionaryOptions;
        bool IsBuildingFinish = false;
    };

    class TUnigramDictionaryBuilderImpl final : public IDictionaryBuilderImpl {
    public:
        TUnigramDictionaryBuilderImpl(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions
        )
            : IDictionaryBuilderImpl(dictionaryBuilderOptions, dictionaryOptions)
        {
            Y_ENSURE(dictionaryOptions.GramOrder > 0);
            Y_ENSURE(dictionaryOptions.GramOrder == 1 || dictionaryOptions.TokenLevelType == ETokenLevelType::Letter);
        }

        void Add(TConstArrayRef<TString> tokens, ui64 weight) override {
            AddImpl(tokens, weight);
        }

        void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) override {
            AddImpl(tokens, weight);
        }

        TIntrusivePtr<TDictionary> FinishBuilding() override;
    private:
        template <typename TTokenType>
        void AddImpl(TConstArrayRef<TTokenType> tokens, ui64 weight);

        NFH::TFlatHashMap<TString, ui64> TokenToCount;

        NFH::TFlatHashMap<TString, TTokenId> TokenToId;
        TVector<TStringBuf> IdToToken;
        TVector<ui64> IdToCount;
    };

    template <ui32 GramOrder>
    class TMultigramDictionaryBuilderImpl final : public IDictionaryBuilderImpl {
    public:
        TMultigramDictionaryBuilderImpl(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions
        )
            : IDictionaryBuilderImpl(dictionaryBuilderOptions, dictionaryOptions)
        {
            Y_ENSURE(dictionaryOptions.GramOrder > 1);
            Y_ENSURE(dictionaryOptions.TokenLevelType == ETokenLevelType::Word);
        }

        void Add(TConstArrayRef<TString> tokens, ui64 weight) override {
            AddImpl(tokens, weight);
        }

        void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) override {
            AddImpl(tokens, weight);
        }

        TIntrusivePtr<TDictionary> FinishBuilding() override;
    private:
        template <typename TTokenType>
        void AddImpl(TConstArrayRef<TTokenType> tokens, ui64 weight);
        void Filter();
        void FilterInternalIdToTokenMapping();

        NFH::TFlatHashMap<TString, TInternalTokenId> TokenToInternalId;
        TInternalIdsMap<GramOrder, ui64> InternalIdsToCount;

        NFH::TFlatHashMap<TInternalTokenId, TStringBuf> InternalIdToToken;
        TInternalIdsMap<GramOrder, TTokenId> InternalIdsToId;
        TVector<const TMultiInternalTokenId<GramOrder>*> IdToInternalIds;
        TVector<ui64> IdToCount;

        bool NeedToFilterInternalIdToTokenMapping = false;
    };

    // TUnigramDictionaryBuilderImpl

    template <typename TTokenType>
    void TUnigramDictionaryBuilderImpl::AddImpl(TConstArrayRef<TTokenType> tokens, ui64 weight) {
        if (DictionaryOptions.TokenLevelType == ETokenLevelType::Word) {
            for (const auto& token : tokens) {
                TokenToCount[token] += weight;
            }
        } else {
            auto updateTokenToCountFunc = [&] (TStringBuf token) {
                TokenToCount[token] += weight;
            };
            ApplyFuncToLetterNGrams(
                tokens,
                DictionaryOptions.GramOrder,
                DictionaryOptions.EndOfWordTokenPolicy == EEndOfWordTokenPolicy::Insert,
                updateTokenToCountFunc
            );
        }
    }

    TIntrusivePtr<TDictionary> TUnigramDictionaryBuilderImpl::FinishBuilding() {
        Y_ENSURE(!IsBuildingFinish, "FinishBuilding method should be called only once.");
        IsBuildingFinish = true;

        TVector<ui64> counts;
        TVector<TString> tokens;
        for (const auto& [key, value] : TokenToCount) {
            if (value < DictionaryBuilderOptions.OccurrenceLowerBound) {
                continue;
            }
            counts.push_back(value);
            tokens.emplace_back(key);
        }

        const ui32 dictionarySize = tokens.size();

        TVector<ui32> indices(dictionarySize);
        Iota(indices.begin(), indices.end(), 0);
        Sort(indices, [&](ui32 lhs, ui32 rhs) {
            return (
                counts[lhs] > counts[rhs] ||
                (counts[lhs] == counts[rhs] && tokens[lhs] < tokens[rhs]
            ));
        });

        const auto maxDictionarySize = GetMaxDictionarySize(DictionaryBuilderOptions.MaxDictionarySize);
        const auto finalDictonarySize = Min(dictionarySize, maxDictionarySize);

        TokenToId.reserve(finalDictonarySize);
        IdToCount.reserve(finalDictonarySize);
        TTokenId globalTokenId = DictionaryOptions.StartTokenId;
        for (ui32 i : xrange(finalDictonarySize)) {
            TokenToId.emplace(tokens[indices[i]], globalTokenId++);
            IdToCount.emplace_back(counts[indices[i]]);
        }

        IdToToken.reserve(finalDictonarySize);
        for (ui32 i : xrange(finalDictonarySize)) {
            IdToToken.emplace_back(TokenToId.find(tokens[indices[i]])->first);
        }

        THolder<IDictionaryImpl> dictionaryImpl = MakeHolder<TUnigramDictionaryImpl>(
            DictionaryOptions,
            std::move(TokenToId),
            std::move(IdToToken),
            std::move(IdToCount)
        );

        return MakeIntrusive<TDictionary>(std::move(dictionaryImpl));
    }

    // TMultigramDictionaryBuilderImpl

    template <ui32 GramOrder>
    static void ShiftAndAddId(TInternalTokenId id, TMultiInternalTokenId<GramOrder>* key) {
        for (ui32 gramIndex = 0; gramIndex < GramOrder - 1; ++gramIndex) {
            (*key)[gramIndex] = (*key)[gramIndex + 1];
        }
        (*key)[GramOrder - 1] = id;
    }

    template <ui32 GramOrder>
    template <typename TTokenType>
    void TMultigramDictionaryBuilderImpl<GramOrder>::AddImpl(TConstArrayRef<TTokenType> rawTokens, ui64 weight) {
        TVector<TTokenType> vectorWithEndOfSentence;
        auto tokens = AppendEndOfSentenceTokenIfNeed(
            rawTokens,
            DictionaryOptions.EndOfSentenceTokenPolicy,
            &vectorWithEndOfSentence
        );

        const auto skipStep = DictionaryOptions.SkipStep;
        const auto tokenCount = tokens.Size();
        TMultiInternalTokenId<GramOrder> key;
        if (skipStep == 0) {
            if (tokenCount < GramOrder) {
                return;
            }

            for (ui32 gramIndex = 0; gramIndex < GramOrder; ++gramIndex) {
                const auto& token = tokens[gramIndex];
                key[gramIndex] = GetInternalWordTokenId(token, &TokenToInternalId);
            }
            InternalIdsToCount[key] += weight;

            for (ui32 tokenIndex = GramOrder; tokenIndex < tokenCount; ++tokenIndex) {
                ShiftAndAddId(GetInternalWordTokenId(tokens[tokenIndex], &TokenToInternalId), &key);
                InternalIdsToCount[key] += weight;
            }
        } else {
            const auto endTokenIndex = GetEndTokenIndex(tokenCount, GramOrder, skipStep);
            for (ui32 tokenIndex = 0; tokenIndex < endTokenIndex; ++tokenIndex) {
                for (ui32 gramIndex = 0; gramIndex < GramOrder; ++gramIndex) {
                    const auto& token = tokens[tokenIndex + gramIndex * (skipStep + 1)];
                    key[gramIndex] = GetInternalWordTokenId(token, &TokenToInternalId);
                }
                InternalIdsToCount[key] += weight;
            }
        }
    }

    template <ui32 GramOrder>
    static bool CompareNGram(
        const TMultiInternalTokenId<GramOrder>& leftNGram,
        const TMultiInternalTokenId<GramOrder>& rightNGram,
        const NFH::TFlatHashMap<TInternalTokenId, TStringBuf>& internalIdToToken
    ) {
        for (ui32 gramIndex = 0; gramIndex < GramOrder; ++gramIndex) {
            if (internalIdToToken.at(leftNGram[gramIndex]) == internalIdToToken.at(rightNGram[gramIndex])) {
                continue;
            }
            return internalIdToToken.at(leftNGram[gramIndex]) < internalIdToToken.at(rightNGram[gramIndex]);
        }
        return false;
    }

    template <ui32 GramOrder>
    void TMultigramDictionaryBuilderImpl<GramOrder>::Filter() {
        TVector<ui64> counts;
        TVector<const TMultiInternalTokenId<GramOrder>*> keys;
        for (const auto& it : InternalIdsToCount) {
            if (it.second < DictionaryBuilderOptions.OccurrenceLowerBound) {
                continue;
            }
            counts.push_back(it.second);
            keys.push_back(&it.first);
        }

        const ui32 dictionarySize = keys.size();

        InternalIdToToken.reserve(TokenToInternalId.size());
        for (const auto& it : TokenToInternalId) {
            InternalIdToToken[it.second] = it.first;
        }

        TVector<ui32> indices(keys.size());
        Iota(indices.begin(), indices.end(), 0);
        Sort(indices, [&](ui32 lhs, ui32 rhs) {
            return (
                counts[lhs] > counts[rhs] ||
                (counts[lhs] == counts[rhs] && CompareNGram(*(keys[lhs]), *(keys[rhs]), InternalIdToToken))
            );
        });

        const auto maxDictionarySize = GetMaxDictionarySize(DictionaryBuilderOptions.MaxDictionarySize);
        const auto finalDictonarySize = Min(dictionarySize, maxDictionarySize);

        if (finalDictonarySize < InternalIdsToCount.size()) {
            NeedToFilterInternalIdToTokenMapping = true;
        }

        TTokenId globalTokenId = DictionaryOptions.StartTokenId;
        InternalIdsToId.reserve(finalDictonarySize);
        IdToCount.reserve(finalDictonarySize);
        for (ui32 i : xrange(finalDictonarySize)) {
            InternalIdsToId[*keys[indices[i]]] = globalTokenId++;
            IdToCount.push_back(counts[indices[i]]);
        }

        IdToInternalIds.reserve(finalDictonarySize);
        for (ui32 i : xrange(finalDictonarySize)) {
            IdToInternalIds.push_back(&(InternalIdsToId.find(*keys[indices[i]])->first));
        }
    }

    template <ui32 GramOrder>
    void TMultigramDictionaryBuilderImpl<GramOrder>::FilterInternalIdToTokenMapping() {
        NFH::TFlatHashMap<TString, TInternalTokenId> tokenToInternalId;
        for (const auto& internalIds : IdToInternalIds) {
            for (ui32 gramIndex : xrange(GramOrder)) {
                TInternalTokenId id = (*internalIds)[gramIndex];
                TStringBuf token = InternalIdToToken.at(id);
                if (!tokenToInternalId.contains(token)) {
                    tokenToInternalId.emplace(token, id);
                }
            }
        }

        TokenToInternalId = std::move(tokenToInternalId);
        InternalIdToToken.clear();
        InternalIdToToken.reserve(TokenToInternalId.size());
        for (const auto& it : TokenToInternalId) {
            InternalIdToToken[it.second] = it.first;
        }
    }

    template <ui32 GramOrder>
    TIntrusivePtr<TDictionary> TMultigramDictionaryBuilderImpl<GramOrder>::FinishBuilding() {
        Y_ENSURE(!IsBuildingFinish, "FinishBuilding method should be called only once.");
        IsBuildingFinish = true;

        Filter();
        if (NeedToFilterInternalIdToTokenMapping) {
            FilterInternalIdToTokenMapping();
        }

        THolder<IDictionaryImpl> dictionaryImpl = MakeHolder<TMultigramDictionaryImpl<GramOrder>>(
            DictionaryOptions,
            std::move(TokenToInternalId),
            std::move(InternalIdsToId),
            std::move(IdToInternalIds),
            std::move(InternalIdToToken),
            std::move(IdToCount)
        );

        return MakeIntrusive<TDictionary>(std::move(dictionaryImpl));
    }

    TDictionaryBuilder::TDictionaryBuilder(TDictionaryBuilder&&) = default;
    TDictionaryBuilder::~TDictionaryBuilder() = default;

    TDictionaryBuilder::TDictionaryBuilder(
        const TDictionaryBuilderOptions& dictionaryBuilderOptions,
        const TDictionaryOptions& dictionaryOptions
    ) {
        Y_ENSURE(dictionaryOptions.GramOrder > 0, "GramOrder should be positive.");
        Y_ENSURE(
            dictionaryOptions.TokenLevelType == ETokenLevelType::Word ||
            dictionaryOptions.SkipStep == 0,
            "SkipStep should be equal to zero in case of Letter token level type."
        );

        if (dictionaryOptions.GramOrder == 1 || dictionaryOptions.TokenLevelType == ETokenLevelType::Letter) {
            DictionaryBuilderImpl = MakeHolder<TUnigramDictionaryBuilderImpl>(dictionaryBuilderOptions, dictionaryOptions);
            return;
        }

        switch (dictionaryOptions.GramOrder) {
            case 2:
                DictionaryBuilderImpl = MakeHolder<TMultigramDictionaryBuilderImpl<2>>(dictionaryBuilderOptions, dictionaryOptions);
                break;
            case 3:
                DictionaryBuilderImpl = MakeHolder<TMultigramDictionaryBuilderImpl<3>>(dictionaryBuilderOptions, dictionaryOptions);
                break;
            case 4:
                DictionaryBuilderImpl = MakeHolder<TMultigramDictionaryBuilderImpl<4>>(dictionaryBuilderOptions, dictionaryOptions);
                break;
            case 5:
                DictionaryBuilderImpl = MakeHolder<TMultigramDictionaryBuilderImpl<5>>(dictionaryBuilderOptions, dictionaryOptions);
                break;
            default:
                Y_ENSURE(false, "Unsupported gram order: " << dictionaryOptions.GramOrder << ".");
        }
    }

    void TDictionaryBuilder::Add(TStringBuf token, ui64 weight) {
        DictionaryBuilderImpl->Add(token, weight);
    }

    void TDictionaryBuilder::Add(TConstArrayRef<TString> tokens, ui64 weight) {
        DictionaryBuilderImpl->Add(tokens, weight);
    }

    void TDictionaryBuilder::Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) {
        DictionaryBuilderImpl->Add(tokens, weight);
    }

    TIntrusivePtr<TDictionary> TDictionaryBuilder::FinishBuilding() {
        return DictionaryBuilderImpl->FinishBuilding();
    }
}
