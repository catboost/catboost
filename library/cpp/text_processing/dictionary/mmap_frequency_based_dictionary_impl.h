#pragma once

#include "fbs_helpers.h"
#include "options.h"
#include "mmap_hash_table.h"
#include "multigram_dictionary_helpers.h"
#include "serialization_helpers.h"

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <library/cpp/containers/flat_hash/flat_hash.h>
#include <library/cpp/text_processing/dictionary/idl/dictionary_meta_info.fbs.h>

#include <util/digest/multi.h>
#include <util/digest/murmur.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/split.h>

#include <array>

namespace NTextProcessing::NDictionary {

    class IMMapDictionaryImpl {
    public:
        IMMapDictionaryImpl() = default;
        explicit IMMapDictionaryImpl(TVector<ui8>&& dictionaryMetaInfoBuffer)
            : DictionaryMetaInfoBuffer(std::move(dictionaryMetaInfoBuffer))
            , DictionaryMetaInfo(NTextProcessingFbs::GetTDictionaryMetaInfo(DictionaryMetaInfoBuffer.data()))
        {
        }

        explicit IMMapDictionaryImpl(const ui8* dictionaryMetaInfoBufferData)
            : DictionaryMetaInfo(NTextProcessingFbs::GetTDictionaryMetaInfo(dictionaryMetaInfoBufferData))
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

        ui32 Size() const {
            return DictionaryMetaInfo->DictionarySize();
        }

        TTokenId GetUnknownTokenId() const {
            return DictionaryMetaInfo->UnknownTokenId();
        }

        TTokenId GetEndOfSentenceTokenId() const {
            return DictionaryMetaInfo->EndOfSentenceTokenId();
        }

        TTokenId GetMinUnusedTokenId() const {
            return DictionaryMetaInfo->EndOfSentenceTokenId() + 1;
        }

        virtual void Save(IOutputStream* stream) const = 0;
        virtual void Load(IInputStream* stream) = 0;

        virtual void InitFromMemory(const ui8* data, size_t size) = 0;

        virtual ~IMMapDictionaryImpl() = default;

    protected:
        TVector<ui8> DictionaryMetaInfoBuffer;
        const NTextProcessingFbs::TDictionaryMetaInfo* DictionaryMetaInfo;
    };

    class TMMapUnigramDictionaryImpl final : public IMMapDictionaryImpl {
    public:
        explicit TMMapUnigramDictionaryImpl(
            TVector<ui8>&& dictionaryMetaInfoBuffer,
            TVector<TBucket>&& tokenToId = {},
            ui64 tokenToIdSeed = 0
        )
            : IMMapDictionaryImpl(std::move(dictionaryMetaInfoBuffer))
            , TokenToIdBuffer(std::move(tokenToId))
            , TokenToId(TokenToIdBuffer)
            , TokenToIdSeed(tokenToIdSeed)
        {
        }

        explicit TMMapUnigramDictionaryImpl(const ui8* dictionaryMetaInfoBufferData)
            : IMMapDictionaryImpl(dictionaryMetaInfoBufferData)
        {
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

        void Save(IOutputStream* stream) const override;
        void Load(IInputStream* stream) override;

        void InitFromMemory(const ui8* data, size_t size) override;

    private:
        template <typename TTokenType>
        void ApplyImpl(
            TConstArrayRef<TTokenType> tokens,
            EUnknownTokenPolicy unknownTokenPolicy,
            TVector<TTokenId>* tokenIds
        ) const;

        TVector<TBucket> TokenToIdBuffer;
        TConstArrayRef<TBucket> TokenToId;
        ui64 TokenToIdSeed = 0;
    };

    template <ui32 GramOrder>
    class TMMapMultigramDictionaryImpl final : public IMMapDictionaryImpl {
    public:
        explicit TMMapMultigramDictionaryImpl(
            TVector<ui8>&& dictionaryMetaInfoBuffer,
            TVector<TBucket>&& tokenToInternalId = {},
            ui64 tokenToInternalIdSeed = 0,
            TVector<TBucket>&& internalIdsToId = {},
            ui64 internalIdsToIdSeed = 0
        )
            : IMMapDictionaryImpl(std::move(dictionaryMetaInfoBuffer))
            , TokenToInternalIdBuffer(std::move(tokenToInternalId))
            , TokenToInternalId(TokenToInternalIdBuffer)
            , TokenToInternalIdSeed(tokenToInternalIdSeed)
            , InternalIdsToIdBuffer(std::move(internalIdsToId))
            , InternalIdsToId(InternalIdsToIdBuffer)
            , InternalIdsToIdSeed(internalIdsToIdSeed)
        {
        }

        explicit TMMapMultigramDictionaryImpl(const ui8* dictionaryMetaInfoBufferData)
            : IMMapDictionaryImpl(dictionaryMetaInfoBufferData)
        {
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


        void Save(IOutputStream* stream) const override {
            stream->Write(MAGIC, MAGIC_SIZE);
            AddPadding(16 - MAGIC_SIZE, stream);

            const ui64 dictionaryMetaInfoBufferSize = DictionaryMetaInfoBuffer.size();
            const ui64 tokenToInternalIdSize = TokenToInternalId.size() * sizeof(TBucket);
            const ui64 internalIdsToIdSize = InternalIdsToId.size() * sizeof(TBucket);
            const ui64 totalSize = 16 + dictionaryMetaInfoBufferSize + 16 + tokenToInternalIdSize + 16 + internalIdsToIdSize;

            WriteLittleEndian(totalSize, stream);
            WriteLittleEndian(dictionaryMetaInfoBufferSize, stream);
            stream->Write(reinterpret_cast<const char*>(DictionaryMetaInfoBuffer.data()), dictionaryMetaInfoBufferSize);

            WriteLittleEndian(tokenToInternalIdSize, stream);
            WriteLittleEndian(TokenToInternalIdSeed, stream);
            stream->Write(reinterpret_cast<const char*>(TokenToInternalId.data()), tokenToInternalIdSize);

            WriteLittleEndian(internalIdsToIdSize, stream);
            WriteLittleEndian(InternalIdsToIdSeed, stream);
            stream->Write(reinterpret_cast<const char*>(InternalIdsToId.data()), internalIdsToIdSize);
        }

        void Load(IInputStream* stream) override {
            ui64 tokenToInternalIdSize;
            ReadLittleEndian(&tokenToInternalIdSize, stream);
            ReadLittleEndian(&TokenToInternalIdSeed, stream);
            TokenToInternalIdBuffer.resize(tokenToInternalIdSize / sizeof(TBucket));
            stream->LoadOrFail(TokenToInternalIdBuffer.data(), tokenToInternalIdSize);
            TokenToInternalId = MakeConstArrayRef(TokenToInternalIdBuffer);

            ui64 internalIdsToIdSize;
            ReadLittleEndian(&internalIdsToIdSize, stream);
            ReadLittleEndian(&InternalIdsToIdSeed, stream);
            InternalIdsToIdBuffer.resize(internalIdsToIdSize / sizeof(TBucket));
            stream->LoadOrFail(InternalIdsToIdBuffer.data(), internalIdsToIdSize);
            InternalIdsToId = MakeConstArrayRef(InternalIdsToIdBuffer);
        }

        void InitFromMemory(const ui8* data, size_t size) override {
            ui64 tokenToInternalIdSize = *reinterpret_cast<const ui64*>(data);
            data += 8;
            TokenToInternalIdSeed = *reinterpret_cast<const ui64*>(data);
            data += 8;
            const TBucket* tokenToInternalIdbucketDataBegin = reinterpret_cast<const TBucket*>(data);
            const TBucket* tokenToInternalIdbucketDataEnd = reinterpret_cast<const TBucket*>(data + tokenToInternalIdSize);
            TokenToInternalId = MakeArrayRef(tokenToInternalIdbucketDataBegin, tokenToInternalIdbucketDataEnd);
            data += tokenToInternalIdSize;

            ui64 internalIdsToIdSize = *reinterpret_cast<const ui64*>(data);
            data += 8;
            InternalIdsToIdSeed = *reinterpret_cast<const ui64*>(data);
            data += 8;
            const TBucket* internalIdsToIdbucketDataBegin = reinterpret_cast<const TBucket*>(data);
            const TBucket* internalIdsToIdbucketDataEnd = reinterpret_cast<const TBucket*>(data + internalIdsToIdSize);
            InternalIdsToId = MakeArrayRef(internalIdsToIdbucketDataBegin, internalIdsToIdbucketDataEnd);
            Y_ENSURE(size == 16 + 16 + tokenToInternalIdSize + internalIdsToIdSize);
        }

    private:
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
                FromFbs(DictionaryMetaInfo->DictionaryOptions()->EndOfSentenceTokenPolicy()),
                &vectorWithEndOfSentence
            );
            const ui32 tokenCount = tokens.Size();

            TVector<TInternalTokenId> internalTokenIds;
            internalTokenIds.reserve(tokenCount);

            for (ui32 tokenIndex = 0; tokenIndex < tokenCount; ++tokenIndex) {
                const auto& token = tokens[tokenIndex];
                auto hash = MurmurHash<ui64>((void*)token.data(), token.size(), TokenToInternalIdSeed);
                const auto& bucket = TokenToInternalId[GetBucketIndex(hash, TokenToInternalId)];
                internalTokenIds.emplace_back(bucket.Hash == hash ? bucket.TokenId : UNKNOWN_INTERNAL_TOKEN_ID);
            }

            const ui32 skipStep = DictionaryMetaInfo->DictionaryOptions()->SkipStep();
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
                    auto hash = MurmurHash<ui64>(
                        (void*)key.data(),
                        sizeof(TInternalTokenId) * GramOrder,
                        InternalIdsToIdSeed
                    );
                    const auto& bucket = InternalIdsToId[GetBucketIndex(hash, InternalIdsToId)];
                    if(bucket.Hash == hash) {
                        tokenIds->push_back(bucket.TokenId);
                        continue;
                    }
                }
                if (unknownTokenPolicy == EUnknownTokenPolicy::Insert) {
                    tokenIds->push_back(DictionaryMetaInfo->UnknownTokenId());
                }
            }
        }

        TVector<TBucket> TokenToInternalIdBuffer;
        TConstArrayRef<TBucket> TokenToInternalId;
        ui64 TokenToInternalIdSeed = 0;
        TVector<TBucket> InternalIdsToIdBuffer;
        TConstArrayRef<TBucket> InternalIdsToId;
        ui64 InternalIdsToIdSeed = 0;
    };

}
