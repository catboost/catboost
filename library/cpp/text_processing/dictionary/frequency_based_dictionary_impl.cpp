#include "frequency_based_dictionary_impl.h"

#include "util.h"

using namespace NTextProcessing::NDictionary;

TTokenId TUnigramDictionaryImpl::Apply(TStringBuf token) const {
    const auto it = TokenToId.find(token);
    return it != TokenToId.end() ? it->second : UnknownTokenId;
}

template <typename TTokenType>
void TUnigramDictionaryImpl::ApplyImpl(
    TConstArrayRef<TTokenType> tokens,
    EUnknownTokenPolicy unknownTokenPolicy,
    TVector<TTokenId>* tokenIds
) const {
    tokenIds->clear();

    auto applyFunc = [&](TStringBuf token) {
        if (const auto it = TokenToId.find(token); it != TokenToId.end()) {
            tokenIds->push_back(it->second);
        } else if (unknownTokenPolicy == EUnknownTokenPolicy::Insert) {
            tokenIds->push_back(UnknownTokenId);
        }
    };

    if (DictionaryOptions.TokenLevelType == ETokenLevelType::Word) {
        for (const auto& token : tokens) {
            applyFunc(token);
        }
        if (DictionaryOptions.EndOfSentenceTokenPolicy == EEndOfSentenceTokenPolicy::Insert) {
            tokenIds->push_back(EndOfSentenceTokenId);
        }
    } else {
        ApplyFuncToLetterNGrams(
            tokens,
            DictionaryOptions.GramOrder,
            DictionaryOptions.EndOfWordTokenPolicy == EEndOfWordTokenPolicy::Insert,
            applyFunc
        );
    }
}

void TUnigramDictionaryImpl::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
}

void TUnigramDictionaryImpl::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
}

ui32 TUnigramDictionaryImpl::Size() const {
    return TokenToId.size();
}

TString TUnigramDictionaryImpl::GetToken(TTokenId tokenId) const {
    if (tokenId == GetEndOfSentenceTokenId()) {
        return "_EOS_";
    } else if (tokenId == GetUnknownTokenId()) {
        return "_UNK_";
    }

    Y_ENSURE(DictionaryOptions.StartTokenId <= tokenId && tokenId < GetMinUnusedTokenId(), "Invalid tokenId.");
    Y_ENSURE(TokenToId.empty() || !IdToToken.empty(), "Internal vector IdToToken is empty.");
    return TString(IdToToken[tokenId - DictionaryOptions.StartTokenId]);
}

ui64 TUnigramDictionaryImpl::GetCount(TTokenId tokenId) const {
    Y_ENSURE(TokenToId.empty() || !IdToCount.empty(), "Internal vector IdToCount is empty.");
    const TTokenId tokenIndex = tokenId - DictionaryOptions.StartTokenId;
    Y_ENSURE(tokenIndex < IdToCount.size(), "Invalid tokenId.");
    return IdToCount[tokenIndex];
}

TVector<TString> TUnigramDictionaryImpl::GetTopTokens(ui32 topSize) const {
    Y_ENSURE(!IdToToken.empty(), "Internal vector IdToToken is empty.");
    const auto correctTopSize = Min<ui32>(topSize, IdToToken.size());
    TVector<TString> result;
    result.reserve(correctTopSize);
    for (auto tokenIndex : xrange(correctTopSize)) {
        result.emplace_back(IdToToken[tokenIndex]);
    }
    return result;
}

void TUnigramDictionaryImpl::ClearStatsData() {
    IdToToken.clear();
    IdToToken.shrink_to_fit();
    IdToCount.clear();
    IdToCount.shrink_to_fit();
}

TTokenId TUnigramDictionaryImpl::GetUnknownTokenId() const {
    return UnknownTokenId;
}

TTokenId TUnigramDictionaryImpl::GetEndOfSentenceTokenId() const {
    return EndOfSentenceTokenId;
}

TTokenId TUnigramDictionaryImpl::GetMinUnusedTokenId() const {
    return EndOfSentenceTokenId + 1;
}

static void GetIdToTokenMapping(
    const NFH::TFlatHashMap<TString, ui32>& tokenToId,
    TVector<TStringBuf>* idToToken
) {
    TVector<ui32> ids;
    ids.reserve(tokenToId.size());
    TVector<TStringBuf> tokens;
    tokens.reserve(tokenToId.size());
    for (const auto& [key, value] : tokenToId) {
        ids.push_back(value);
        tokens.emplace_back(key);
    }

    TVector<ui32> indices(ids.size());
    Iota(indices.begin(), indices.end(), 0);
    Sort(indices, [&](ui32 lhs, ui32 rhs) {
        return (ids[lhs] < ids[rhs]);
    });

    idToToken->clear();
    idToToken->reserve(tokens.size());
    for (ui32 i : indices) {
        idToToken->push_back(tokens[i]);
    }
}

void TUnigramDictionaryImpl::Save(IOutputStream* stream) const {
    const ui32 dictionarySize = TokenToId.size();
    TVector<TStringBuf> idToToken;
    if (IdToToken.empty()) {
        GetIdToTokenMapping(TokenToId, &idToToken);
    }
    const TVector<TStringBuf>& idToTokenRef = IdToToken.empty() ? idToToken : IdToToken;
    const bool doSaveCounts = !IdToCount.empty();

    auto dictionaryOptionsJson = DictionaryOptionsToJson(DictionaryOptions);
    dictionaryOptionsJson[DICT_FORMAT_KEY] = DICT_NEW_FORMAT_DESC;
    *stream << dictionaryOptionsJson << '\n';
    *stream << dictionarySize << '\n';
    for (auto tokenIndex : xrange(idToTokenRef.size())) {
        auto token = idToTokenRef[tokenIndex];
        Y_ENSURE(!Count(token, '\n'), TString::Join(
            "It is impossible to save the dictionary because the token '", token, "' has \\n symbol."
        ));
        const auto it = TokenToId.find(token);
        *stream << it->second << '\t';
        if (doSaveCounts) {
            *stream << IdToCount[tokenIndex];
        }
        *stream << '\t' << token << '\n';
    }
}

static void GetTokenInfoFromLineInOldFormat(
    const TString& line,
    NFH::TFlatHashMap<TString, TTokenId>* tokenToId,
    TVector<TStringBuf>* idToToken,
    TVector<ui64>* idToCount
) {
    TVector<TStringBuf> splittedLine;
    StringSplitter(line).Split('\t').Collect(&splittedLine);
    auto token = splittedLine[1];
    (*tokenToId)[token] = FromString<ui32>(splittedLine[0]);
    idToToken->emplace_back(tokenToId->find(token)->first);
    if (splittedLine.size() == 3) {
        idToCount->emplace_back(FromString<ui64>(splittedLine[2]));
    }
}

static void GetTokenInfoFromLineInNewFormat(
    const TString& line,
    NFH::TFlatHashMap<TString, TTokenId>* tokenToId,
    TVector<TStringBuf>* idToToken,
    TVector<ui64>* idToCount
) {
    TVector<TStringBuf> splittedLine;
    StringSplitter(line).Split('\t').Limit(3).Collect(&splittedLine);
    auto token = splittedLine[2];
    (*tokenToId)[token] = FromString<ui32>(splittedLine[0]);
    idToToken->emplace_back(tokenToId->find(token)->first);
    if (!splittedLine[1].empty()) {
        idToCount->emplace_back(FromString<ui64>(splittedLine[1]));
    }
}

void TUnigramDictionaryImpl::Load(IInputStream* stream, bool isNewFormat) {
    const ui32 dictionarySize = FromString<ui32>(stream->ReadLine());

    TokenToId.clear();
    TokenToId.reserve(dictionarySize);
    IdToToken.clear();
    IdToToken.reserve(dictionarySize);
    IdToCount.clear();
    IdToCount.reserve(dictionarySize);

    for (ui32 tokenIndex = 0; tokenIndex < dictionarySize; ++tokenIndex) {
        const TString line = stream->ReadLine();
        if (isNewFormat) {
            GetTokenInfoFromLineInNewFormat(line, &TokenToId, &IdToToken, &IdToCount);
        } else {
            GetTokenInfoFromLineInOldFormat(line, &TokenToId, &IdToToken, &IdToCount);
        }
    }
    IdToCount.shrink_to_fit();
    InitializeSpecialTokenIds();
}

THolder<IMMapDictionaryImpl> TUnigramDictionaryImpl::CreateMMapDictionaryImpl() const {
    TVector<TStringBuf> idToToken;
    if (IdToToken.empty()) {
        GetIdToTokenMapping(TokenToId, &idToToken);
    }
    const TVector<TStringBuf>& idToTokenRef = IdToToken.empty() ? idToToken : IdToToken;

    TVector<TBucket> buckets;
    ui64 seed;
    BuildBuckets(
        xrange<TTokenId>(idToTokenRef.size()),
        [&](TTokenId tokenId, ui64 seed) {
            auto hash = MurmurHash<ui64>(
                (void*)(idToTokenRef[tokenId].data()),
                idToTokenRef[tokenId].size(),
                seed
            );
            return std::make_pair(hash, tokenId);
        },
        &buckets,
        &seed
    );

    TVector<ui8> dictionaryMetaInfoBuffer;
    BuildDictionaryMetaInfo(Size(), DictionaryOptions, &dictionaryMetaInfoBuffer);

    return MakeHolder<TMMapUnigramDictionaryImpl>(
        std::move(dictionaryMetaInfoBuffer),
        std::move(buckets),
        seed
    );
}
