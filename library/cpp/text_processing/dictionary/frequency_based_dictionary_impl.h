#pragma once

#include "fbs_helpers.h"
#include "options.h"
#include "multigram_dictionary_helpers.h"
#include "mmap_frequency_based_dictionary.h"
#include "mmap_frequency_based_dictionary_impl.h"
#include "mmap_hash_table.h"

#include <util/digest/multi.h>
#include <util/digest/murmur.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/string/split.h>

#include <array>

namespace NTextProcessing::NDictionary {

    class IDictionaryImpl {
    public:
        IDictionaryImpl(const TDictionaryOptions& dictionaryOptions)
            : DictionaryOptions(dictionaryOptions)
        {
        }

        virtual TTokenId Apply(const TStringBuf token) const = 0;

        virtual void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const = 0;
        virtual void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const = 0;

        virtual ui32 Size() const = 0;

        virtual TString GetToken(TTokenId tokenId) const = 0;
        virtual ui64 GetCount(TTokenId tokenId) const = 0;
        virtual TVector<TString> GetTopTokens(ui32 topSize = 10) const = 0;

        virtual void ClearStatsData() = 0;

        virtual TTokenId GetUnknownTokenId() const = 0;
        virtual TTokenId GetEndOfSentenceTokenId() const = 0;
        virtual TTokenId GetMinUnusedTokenId() const = 0;

        const TDictionaryOptions& GetDictionaryOptionsRef() const {
            return DictionaryOptions;
        }

        virtual void Save(IOutputStream* stream) const = 0;
        virtual void Load(IInputStream* stream, bool isNewFormat) = 0;

        virtual THolder<IMMapDictionaryImpl> CreateMMapDictionaryImpl() const = 0;

        virtual ~IDictionaryImpl() = default;

    protected:
        TDictionaryOptions DictionaryOptions;

        TTokenId UnknownTokenId;
        TTokenId EndOfSentenceTokenId;
    };

    class TUnigramDictionaryImpl final : public IDictionaryImpl {
    public:

        TUnigramDictionaryImpl() = default;
        explicit TUnigramDictionaryImpl(
            const TDictionaryOptions& dictionaryOptions,
            NFH::TFlatHashMap<TString, TTokenId> tokenToId = {},
            TVector<TStringBuf> idToToken = {},
            TVector<ui64> idToCount = {}
        )
            : IDictionaryImpl(dictionaryOptions)
            , TokenToId(std::move(tokenToId))
            , IdToToken(std::move(idToToken))
            , IdToCount(std::move(idToCount))
        {
            Y_ENSURE(IdToToken.size() == 0 || TokenToId.size() == IdToToken.size(),
                "Tokens count should be equal to dictionary size.");
            InitializeSpecialTokenIds();
        }

        TTokenId Apply(const TStringBuf token) const override;

        void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const override;
        void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const override;

        ui32 Size() const override;

        TString GetToken(TTokenId tokenId) const override;
        ui64 GetCount(TTokenId tokenId) const override;
        TVector<TString> GetTopTokens(ui32 topSize = 10) const override;

        void ClearStatsData() override;

        TTokenId GetUnknownTokenId() const override;
        TTokenId GetEndOfSentenceTokenId() const override;
        TTokenId GetMinUnusedTokenId() const override;

        void Save(IOutputStream* stream) const override;
        void Load(IInputStream* stream, bool isNewFormat) override;

        THolder<IMMapDictionaryImpl> CreateMMapDictionaryImpl() const override;

    private:
        void InitializeSpecialTokenIds() {
            UnknownTokenId = TokenToId.size() + DictionaryOptions.StartTokenId;
            EndOfSentenceTokenId = UnknownTokenId + 1;
        }

        template <typename TTokenType>
        void ApplyImpl(
            TConstArrayRef<TTokenType> tokens,
            EUnknownTokenPolicy unknownTokenPolicy,
            TVector<TTokenId>* tokenIds
        ) const;

        NFH::TFlatHashMap<TString, TTokenId> TokenToId;

        // Optional fields (May be empty)
        TVector<TStringBuf> IdToToken;
        TVector<ui64> IdToCount;
    };

    template <ui32 GramOrder>
    class TMultigramDictionaryImpl final : public IDictionaryImpl {
    public:
        explicit TMultigramDictionaryImpl(
            const TDictionaryOptions& dictionaryOptions,
            NFH::TFlatHashMap<TString, TInternalTokenId> tokenToInternalId = {},
            TInternalIdsMap<GramOrder, TTokenId> internalIdsToId = {},
            TVector<const TMultiInternalTokenId<GramOrder>*> idToInternalIds = {},
            NFH::TFlatHashMap<TInternalTokenId, TStringBuf> internalIdToToken = {},
            TVector<ui64> idToCount = {}
        )
            : IDictionaryImpl(dictionaryOptions)
            , TokenToInternalId(std::move(tokenToInternalId))
            , InternalIdsToId(std::move(internalIdsToId))
            , IdToInternalIds(std::move(idToInternalIds))
            , InternalIdToToken(std::move(internalIdToToken))
            , IdToCount(std::move(idToCount))
        {
            Y_ENSURE(IdToInternalIds.size() == 0 || InternalIdsToId.size() == IdToInternalIds.size());
            Y_ENSURE(InternalIdToToken.size() == 0 || TokenToInternalId.size() == InternalIdToToken.size());
            InitializeSpecialTokenIds();
        }

        TTokenId Apply(const TStringBuf /*token*/) const override {
            Y_ENSURE(false, "Unimplemented for Word Multigram dictionary.");
        }

        void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const override {
            ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
        }

        void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) const override {
            ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
        }

        ui32 Size() const override {
            return InternalIdsToId.size();
        }

        TString GetToken(TTokenId tokenId) const override {
            if (tokenId == GetEndOfSentenceTokenId()) {
                return "_EOS_";
            } else if (tokenId == GetUnknownTokenId()) {
                return "_UNK_";
            }

            Y_ENSURE(DictionaryOptions.StartTokenId <= tokenId && tokenId < GetMinUnusedTokenId(), "Invalid tokenId.");
            Y_ENSURE(!IdToInternalIds.empty(), "Internal vector IdToInternalIds is empty.");

            TTokenId tokenIndex = tokenId - DictionaryOptions.StartTokenId;
            const auto& multiInternalId = *IdToInternalIds[tokenIndex];
            TString result(InternalIdToToken.at(multiInternalId[0]));
            for (ui32 gramIndex : xrange<ui32>(1, GramOrder)) {
                result += " " + ToString(InternalIdToToken.at(multiInternalId[gramIndex]));
            }
            return result;
        }

        ui64 GetCount(TTokenId tokenId) const override {
            const TTokenId tokenIndex = tokenId - DictionaryOptions.StartTokenId;
            Y_ENSURE(tokenIndex < IdToCount.size(), "Invalid tokenId.");
            Y_ENSURE(!IdToCount.empty(), "Internal vector IdToCount is empty.");
            return IdToCount[tokenIndex];
        }

        TVector<TString> GetTopTokens(ui32 topSize = 10) const override {
            Y_ENSURE(!IdToInternalIds.empty(), "Internal vector IdToInternalIds is empty.");
            Y_ENSURE(!InternalIdToToken.empty(), "Internal vector InternalIdToToken is empty.");
            auto correctTopSize = Min<ui32>(topSize, IdToInternalIds.size());
            TVector<TString> result;
            result.reserve(correctTopSize);
            for (auto tokenIndex : xrange(correctTopSize)) {
                const auto& multiInternalId = *IdToInternalIds[tokenIndex];
                TString multiToken(InternalIdToToken.at(multiInternalId[0]));
                for (ui32 gramIndex : xrange<ui32>(1, GramOrder)) {
                    multiToken += " " + ToString(InternalIdToToken.at(multiInternalId[gramIndex]));
                }
                result.push_back(multiToken);
            }
            return result;
        }

        void ClearStatsData() override {
            IdToInternalIds.clear();
            IdToInternalIds.shrink_to_fit();
            InternalIdToToken.clear();
            IdToCount.clear();
            IdToCount.shrink_to_fit();
        }

        TTokenId GetUnknownTokenId() const override {
            return UnknownTokenId;
        }

        TTokenId GetEndOfSentenceTokenId() const override {
            return EndOfSentenceTokenId;
        }

        TTokenId GetMinUnusedTokenId() const override {
            return EndOfSentenceTokenId + 1;
        }

        void Save(IOutputStream* stream) const override {
            const ui32 dictionarySize = InternalIdsToId.size();
            TVector<const TMultiInternalTokenId<GramOrder>*> idToInternalIds;
            NFH::TFlatHashMap<TInternalTokenId, TStringBuf> internalIdToToken;
            if (IdToInternalIds.empty() || InternalIdToToken.empty()) {
                GetIdToTokensMapping(&idToInternalIds, &internalIdToToken);
            }
            const auto& idToInternalIdsRef = IdToInternalIds.empty() ? idToInternalIds : IdToInternalIds;
            const auto& internalIdToTokenRef = InternalIdToToken.empty() ? internalIdToToken : InternalIdToToken;
            const bool doSaveCounts = !IdToCount.empty();

            auto dictionaryOptionsJson = DictionaryOptionsToJson(DictionaryOptions);
            dictionaryOptionsJson[DICT_FORMAT_KEY] = DICT_NEW_FORMAT_DESC;
            *stream << dictionaryOptionsJson << '\n';
            *stream << dictionarySize << '\n';
            for (auto tokenIndex : xrange(idToInternalIdsRef.size())) {
                const auto& internalIds = *(idToInternalIdsRef[tokenIndex]);
                *stream << InternalIdsToId.at(internalIds) << '\t';
                if (doSaveCounts) {
                    *stream << IdToCount[tokenIndex];
                }
                *stream << '\t' << internalIdToTokenRef.at(internalIds[0]);
                for (ui32 gramIndex : xrange<ui32>(1, GramOrder)) {
                    const auto& token = internalIdToTokenRef.at(internalIds[gramIndex]);
                    Y_ENSURE(!Count(token, '\n'), TString::Join(
                        "It is impossible to save the dictionary because the token '", token, "' has \\n symbol."
                    ));
                    Y_ENSURE(!Count(token, ' '), TString::Join(
                        "It is impossible to save the dictionary because the token '", token, "' has space symbol."
                    ));
                    *stream << ' ' << token;
                }
                *stream << '\n';
            }
        }

        static void GetTokenInfoFromLineInOldFormat(
            const TString& line,
            TVector<TMultiInternalTokenId<GramOrder>>* sortedInternalIds,
            NFH::TFlatHashMap<TString, TInternalTokenId>* tokenToInternalId,
            TInternalIdsMap<GramOrder, TTokenId>* internalIdsToId,
            TVector<ui64>* idToCount
        ) {
            TVector<TStringBuf> splittedLine;
            StringSplitter(line).Split('\t').Collect(&splittedLine);

            auto& key = sortedInternalIds->emplace_back();
            ui32 gramIndex = 0;
            for (TStringBuf subToken : StringSplitter(splittedLine[1]).Split(' ')) {
                key[gramIndex++] = GetInternalWordTokenId(subToken, tokenToInternalId);
            }
            internalIdsToId->emplace(key, FromString<ui32>(splittedLine[0]));
            if (splittedLine.size() == 3) {
                idToCount->emplace_back(FromString<ui64>(splittedLine[2]));
            }
        }

        static void GetTokenInfoFromLineInNewFormat(
            const TString& line,
            TVector<TMultiInternalTokenId<GramOrder>>* sortedInternalIds,
            NFH::TFlatHashMap<TString, TInternalTokenId>* tokenToInternalId,
            TInternalIdsMap<GramOrder, TTokenId>* internalIdsToId,
            TVector<ui64>* idToCount
        ) {
            TVector<TStringBuf> splittedLine;
            StringSplitter(line).Split('\t').Limit(3).Collect(&splittedLine);

            auto& key = sortedInternalIds->emplace_back();
            ui32 gramIndex = 0;
            for (TStringBuf subToken : StringSplitter(splittedLine[2]).Split(' ')) {
                key[gramIndex++] = GetInternalWordTokenId(subToken, tokenToInternalId);
            }
            internalIdsToId->emplace(key, FromString<ui32>(splittedLine[0]));
            if (!splittedLine[1].empty()) {
                idToCount->emplace_back(FromString<ui64>(splittedLine[1]));
            }
        }

        void Load(IInputStream* stream, bool isNewFormat) override {
            ui32 dictionarySize = FromString<ui32>(stream->ReadLine());

            InternalIdsToId.clear();
            InternalIdsToId.reserve(dictionarySize);
            IdToInternalIds.clear();
            IdToInternalIds.reserve(dictionarySize);
            IdToCount.clear();
            IdToCount.reserve(dictionarySize);

            TVector<TMultiInternalTokenId<GramOrder>> sortedInternalIds;
            sortedInternalIds.reserve(dictionarySize);

            for (ui32 tokenIndex = 0; tokenIndex < dictionarySize; ++tokenIndex) {
                const TString line = stream->ReadLine();
                if (isNewFormat) {
                    GetTokenInfoFromLineInNewFormat(line, &sortedInternalIds, &TokenToInternalId, &InternalIdsToId, &IdToCount);
                } else {
                    GetTokenInfoFromLineInOldFormat(line, &sortedInternalIds, &TokenToInternalId, &InternalIdsToId, &IdToCount);
                }

            }
            IdToCount.shrink_to_fit();

            for (const auto& internalIds : sortedInternalIds) {
                const auto it = InternalIdsToId.find(internalIds);
                IdToInternalIds.push_back(&it->first);
            }

            InternalIdToToken.reserve(TokenToInternalId.size());
            for (const auto& it : TokenToInternalId) {
                InternalIdToToken[it.second] = it.first;
            }

            InitializeSpecialTokenIds();
        }

        THolder<IMMapDictionaryImpl> CreateMMapDictionaryImpl() const override {
            TVector<TBucket> tokenToInternalIdBuckets;
            ui64 tokenToInternalIdBucketsSeed;
            BuildBuckets(
                TokenToInternalId,
                [](const auto& it, ui64 seed) {
                    auto hash = MurmurHash<ui64>((void*)(it.first.data()), it.first.size(), seed);
                    return std::make_pair(hash, it.second);
                },
                &tokenToInternalIdBuckets,
                &tokenToInternalIdBucketsSeed
            );

            TVector<const TMultiInternalTokenId<GramOrder>*> idToInternalIds;
            NFH::TFlatHashMap<TInternalTokenId, TStringBuf> internalIdToToken;
            if (IdToInternalIds.empty() || InternalIdToToken.empty()) {
                GetIdToTokensMapping(&idToInternalIds, &internalIdToToken);
            }
            const auto& idToInternalIdsRef = IdToInternalIds.empty() ? idToInternalIds : IdToInternalIds;

            TVector<TBucket> internalIdsToIdBuckets;
            ui64 internalIdsToIdBucketsSeed;
            BuildBuckets(
                xrange(idToInternalIdsRef.size()),
                [&](TTokenId tokenId, ui64 seed) {
                    auto hash = MurmurHash<ui64>(
                        (void*)(idToInternalIdsRef[tokenId]->data()),
                        sizeof(TInternalTokenId) * GramOrder,
                        seed
                    );
                    return std::make_pair(hash, tokenId);
                },
                &internalIdsToIdBuckets,
                &internalIdsToIdBucketsSeed
            );

            TVector<ui8> dictionaryMetaInfoBuffer;
            BuildDictionaryMetaInfo(Size(), DictionaryOptions, &dictionaryMetaInfoBuffer);

            return MakeHolder<TMMapMultigramDictionaryImpl<GramOrder>>(
                std::move(dictionaryMetaInfoBuffer),
                std::move(tokenToInternalIdBuckets),
                tokenToInternalIdBucketsSeed,
                std::move(internalIdsToIdBuckets),
                internalIdsToIdBucketsSeed
            );
        }

    private:
        void InitializeSpecialTokenIds() {
            UnknownTokenId = InternalIdsToId.size() + DictionaryOptions.StartTokenId;
            EndOfSentenceTokenId = UnknownTokenId + 1;
        }

        template <typename TTokenType>
        void ApplyImpl(
            TConstArrayRef<TTokenType> rawTokens,
            EUnknownTokenPolicy unknownTokenPolicy,
            TVector<TTokenId>* tokenIds
        ) const {
            tokenIds->clear();

            TVector<TTokenType> vectorWithEndOfSentence;
            auto tokens = AppendEndOfSentenceTokenIfNeed(
                rawTokens,
                DictionaryOptions.EndOfSentenceTokenPolicy,
                &vectorWithEndOfSentence
            );
            const ui32 tokenCount = tokens.Size();

            TVector<TInternalTokenId> internalTokenIds;
            internalTokenIds.reserve(tokenCount);

            for (ui32 tokenIndex = 0; tokenIndex < tokenCount; ++tokenIndex) {
                const auto it = TokenToInternalId.find(tokens[tokenIndex]);
                if (it != TokenToInternalId.end()) {
                    internalTokenIds.emplace_back(it->second);
                } else {
                    internalTokenIds.emplace_back(UNKNOWN_INTERNAL_TOKEN_ID);
                }
            }

            const auto skipStep = DictionaryOptions.SkipStep;
            TMultiInternalTokenId<GramOrder> key;

            const auto endTokenIndex = GetEndTokenIndex(tokens.Size(), GramOrder, skipStep);

            for (ui32 tokenIndex = 0; tokenIndex < endTokenIndex; ++tokenIndex) {
                bool hasUnknownInternalTokenId = false;
                for (ui32 gramIndex = 0; gramIndex < GramOrder; ++gramIndex) {
                    const auto internalTokenId = internalTokenIds[tokenIndex + gramIndex * (skipStep + 1)];
                    if (internalTokenId != UNKNOWN_INTERNAL_TOKEN_ID) {
                        key[gramIndex] = internalTokenId;
                    } else {
                        hasUnknownInternalTokenId = true;
                        break;
                    }
                }
                if (!hasUnknownInternalTokenId) {
                    const auto it = InternalIdsToId.find(key);
                    if (it != InternalIdsToId.end()) {
                        tokenIds->push_back(it->second);
                        continue;
                    }
                }
                if (unknownTokenPolicy == EUnknownTokenPolicy::Insert) {
                    tokenIds->push_back(UnknownTokenId);
                }
            }
        }

        void GetIdToTokensMapping(
            TVector<const TMultiInternalTokenId<GramOrder>*>* idToInternalIds,
            NFH::TFlatHashMap<TInternalTokenId, TStringBuf>* internalIdToToken
        ) const {
            const ui32 dictionarySize = InternalIdsToId.size();

            TVector<TTokenId> tokenIds;
            tokenIds.reserve(dictionarySize);
            TVector<const TMultiInternalTokenId<GramOrder>*> keys;
            keys.reserve(dictionarySize);
            for (const auto& it : InternalIdsToId) {
                tokenIds.push_back(it.second);
                keys.push_back(&it.first);
            }

            internalIdToToken->reserve(TokenToInternalId.size());
            for (const auto& it : TokenToInternalId) {
                (*internalIdToToken)[it.second] = it.first;
            }

            TVector<ui32> indices(tokenIds.size());
            Iota(indices.begin(), indices.end(), 0);
            Sort(indices, [&](ui32 lhs, ui32 rhs) {
                return tokenIds[lhs] < tokenIds[rhs];
            });

            idToInternalIds->reserve(dictionarySize);
            for (ui32 i : indices) {
                idToInternalIds->push_back(keys[i]);
            }
        }

        NFH::TFlatHashMap<TString, TInternalTokenId> TokenToInternalId;
        TInternalIdsMap<GramOrder, TTokenId> InternalIdsToId;

        // Optional fields (May be empty)
        TVector<const TMultiInternalTokenId<GramOrder>*> IdToInternalIds;
        NFH::TFlatHashMap<TInternalTokenId, TStringBuf> InternalIdToToken;
        TVector<ui64> IdToCount;
    };
}

template<ui32 N>
struct THash<NTextProcessing::NDictionary::TMultiInternalTokenId<N>> {
    size_t operator()(const NTextProcessing::NDictionary::TMultiInternalTokenId<N>& arr) const {
        size_t result = arr[0];
        for (size_t i = 1; i < N; ++i) {
            result = MultiHash(result, arr[i]);
        }
        return result;
    }
};
