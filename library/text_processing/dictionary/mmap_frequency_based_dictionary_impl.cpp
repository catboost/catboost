#include "mmap_frequency_based_dictionary_impl.h"
#include "util.h"

using namespace NTextProcessing::NDictionary;

TTokenId TMMapUnigramDictionaryImpl::Apply(TStringBuf token) const {
    auto hash = MurmurHash<ui64>((void*)token.data(), token.size(), TokenToIdSeed);
    const auto& bucket = TokenToId[GetBucketIndex(hash, TokenToId)];
    return bucket.Hash == hash ? bucket.TokenId : DictionaryMetaInfo->UnknownTokenId();
}

template <typename TTokenType>
void TMMapUnigramDictionaryImpl::ApplyImpl(
    TConstArrayRef<TTokenType> tokens,
    EUnknownTokenPolicy unknownTokenPolicy,
    TVector<TTokenId>* tokenIds
) const {
    tokenIds->clear();

    auto applyFunc = [&](TStringBuf token) {
        auto hash = MurmurHash<ui64>((void*)token.data(), token.size(), TokenToIdSeed);
        const auto& bucket = TokenToId[GetBucketIndex(hash, TokenToId)];
        if (bucket.Hash == hash) {
            tokenIds->push_back(bucket.TokenId);
        } else if (unknownTokenPolicy == EUnknownTokenPolicy::Insert) {
            tokenIds->push_back(DictionaryMetaInfo->UnknownTokenId());
        }
    };

    if (FromFbs(DictionaryMetaInfo->DictionaryOptions()->TokenLevelType()) == ETokenLevelType::Word) {
        for (const auto& token : tokens) {
            applyFunc(token);
        }
        if (FromFbs(DictionaryMetaInfo->DictionaryOptions()->EndOfSentenceTokenPolicy()) == EEndOfSentenceTokenPolicy::Insert) {
            tokenIds->push_back(DictionaryMetaInfo->EndOfSentenceTokenId());
        }
    } else {
        ApplyFuncToLetterNGrams(
            tokens,
            DictionaryMetaInfo->DictionaryOptions()->GramOrder(),
            FromFbs(DictionaryMetaInfo->DictionaryOptions()->EndOfWordTokenPolicy()) == EEndOfWordTokenPolicy::Insert,
            applyFunc
        );
    }
}

void TMMapUnigramDictionaryImpl::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
}

void TMMapUnigramDictionaryImpl::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(tokens, unknownTokenPolicy, tokenIds);
}

void TMMapUnigramDictionaryImpl::Save(IOutputStream* stream) const {
    stream->Write(MAGIC, MAGIC_SIZE);
    AddPadding(16 - MAGIC_SIZE, stream);

    const ui64 dictionaryMetaInfoBufferSize = DictionaryMetaInfoBuffer.size();
    const ui64 tokenToIdSize = TokenToId.size() * sizeof(TBucket);
    const ui64 totalSize = 16 + dictionaryMetaInfoBufferSize + 16 + tokenToIdSize;

    WriteLittleEndian(totalSize, stream);
    WriteLittleEndian(dictionaryMetaInfoBufferSize, stream);
    stream->Write(reinterpret_cast<const char*>(DictionaryMetaInfoBuffer.data()), dictionaryMetaInfoBufferSize);

    WriteLittleEndian(tokenToIdSize, stream);
    WriteLittleEndian(TokenToIdSeed, stream);
    stream->Write(reinterpret_cast<const char*>(TokenToId.data()), tokenToIdSize);
}

void TMMapUnigramDictionaryImpl::Load(IInputStream* stream) {
    ui64 tokenToIdSize;
    ReadLittleEndian(&tokenToIdSize, stream);
    ReadLittleEndian(&TokenToIdSeed, stream);
    TokenToIdBuffer.resize(tokenToIdSize / sizeof(TBucket));
    stream->LoadOrFail(TokenToIdBuffer.data(), tokenToIdSize);
    TokenToId = MakeConstArrayRef(TokenToIdBuffer);
}

void TMMapUnigramDictionaryImpl::InitFromMemory(const ui8* data, size_t size) {
    ui64 tokenToIdSize = *reinterpret_cast<const ui64*>(data);
    data += 8;
    TokenToIdSeed = *reinterpret_cast<const ui64*>(data);
    data += 8;
    const TBucket* bucketDataBegin = reinterpret_cast<const TBucket*>(data);
    const TBucket* bucketDataEnd = reinterpret_cast<const TBucket*>(data + tokenToIdSize);
    TokenToId = MakeArrayRef(bucketDataBegin, bucketDataEnd);
    Y_ENSURE(size == 16 + tokenToIdSize);
}
