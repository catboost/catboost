#include "frequency_based_dictionary_impl.h"
#include "mmap_frequency_based_dictionary.h"
#include "mmap_frequency_based_dictionary_impl.h"
#include "util.h"

#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>

using namespace NTextProcessing::NDictionary;

TMMapDictionary::TMMapDictionary() = default;

TMMapDictionary::TMMapDictionary(TIntrusiveConstPtr<TDictionary> dictionary)
    : DictionaryImpl(dictionary->DictionaryImpl->CreateMMapDictionaryImpl())
{
}

TMMapDictionary::TMMapDictionary(const void* data, size_t size) {
    InitFromMemory(data, size);
}

TMMapDictionary::TMMapDictionary(TMMapDictionary&&) = default;

TMMapDictionary::~TMMapDictionary() = default;

TTokenId TMMapDictionary::Apply(TStringBuf token) const {
    return DictionaryImpl->Apply(token);
}

void TMMapDictionary::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    DictionaryImpl->Apply(tokens, tokenIds, unknownTokenPolicy);
}

void TMMapDictionary::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    DictionaryImpl->Apply(tokens, tokenIds, unknownTokenPolicy);
}

ui32 TMMapDictionary::Size() const {
    return DictionaryImpl->Size();
}

TString TMMapDictionary::GetToken(TTokenId) const {
    Y_ENSURE(false, "Unsupported method");
}

ui64 TMMapDictionary::GetCount(TTokenId) const {
    Y_ENSURE(false, "Unsupported method");
}

TVector<TString> TMMapDictionary::GetTopTokens(ui32) const {
    Y_ENSURE(false, "Unsupported method");
}

void TMMapDictionary::ClearStatsData() {
    Y_ENSURE(false, "Unsupported method");
}

TTokenId TMMapDictionary::GetUnknownTokenId() const {
    return DictionaryImpl->GetUnknownTokenId();
}

TTokenId TMMapDictionary::GetEndOfSentenceTokenId() const {
    return DictionaryImpl->GetEndOfSentenceTokenId();
}

TTokenId TMMapDictionary::GetMinUnusedTokenId() const {
    return DictionaryImpl->GetMinUnusedTokenId();
}

const TDictionaryOptions& TMMapDictionary::GetDictionaryOptionsRef() const {
    Y_ENSURE(false, "Unsupported method");
}

void TMMapDictionary::Save(IOutputStream* stream) const {
    DictionaryImpl->Save(stream);
}

void TMMapDictionary::Load(IInputStream* stream) {
    char magic[MAGIC_SIZE];
    stream->LoadOrFail(magic, MAGIC_SIZE);
    Y_ENSURE(!std::memcmp(magic, MAGIC, MAGIC_SIZE));
    SkipPadding(16 - MAGIC_SIZE, stream);

    ui64 totalSize;;
    ui64 dictionaryMetaInfoBufferSize;
    ReadLittleEndian(&totalSize, stream);
    ReadLittleEndian(&dictionaryMetaInfoBufferSize, stream);
    TVector<ui8> dictionaryMetaInfoBuffer(dictionaryMetaInfoBufferSize);
    stream->LoadOrFail(dictionaryMetaInfoBuffer.data(), dictionaryMetaInfoBufferSize);

    const auto* dictionaryMetaInfo= NTextProcessingFbs::GetTDictionaryMetaInfo(dictionaryMetaInfoBuffer.data());
    const auto tokenLevelType = FromFbs(dictionaryMetaInfo->DictionaryOptions()->TokenLevelType());
    const ui32 gramOrder = dictionaryMetaInfo->DictionaryOptions()->GramOrder();

    if (tokenLevelType == ETokenLevelType::Letter || gramOrder == 1) {
        DictionaryImpl = MakeHolder<TMMapUnigramDictionaryImpl>(std::move(dictionaryMetaInfoBuffer));
    } else {
        switch (gramOrder) {
            case 2:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<2>>(std::move(dictionaryMetaInfoBuffer));
                break;
            case 3:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<3>>(std::move(dictionaryMetaInfoBuffer));
                break;
            case 4:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<4>>(std::move(dictionaryMetaInfoBuffer));
                break;
            case 5:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<5>>(std::move(dictionaryMetaInfoBuffer));
                break;
            default:
                Y_ENSURE(false, "Unsupported gram order: " << gramOrder << ".");
        }
    }

    DictionaryImpl->Load(stream);
}

void TMMapDictionary::InitFromMemory(const void* data, size_t size) {
    const ui8* ptr = reinterpret_cast<const ui8*>(data);
    Y_ENSURE(!std::memcmp(ptr, MAGIC, MAGIC_SIZE));
    ptr += 16;
    const ui64 totalSize = *reinterpret_cast<const ui64*>(ptr);
    ptr += 8;
    const ui64 dictionaryMetaInfoBufferSize = *reinterpret_cast<const ui64*>(ptr);
    ptr += 8;
    const auto* dictionaryMetaInfo= NTextProcessingFbs::GetTDictionaryMetaInfo(ptr);
    const auto tokenLevelType = FromFbs(dictionaryMetaInfo->DictionaryOptions()->TokenLevelType());
    const ui32 gramOrder = dictionaryMetaInfo->DictionaryOptions()->GramOrder();

    if (tokenLevelType == ETokenLevelType::Letter || gramOrder == 1) {
        DictionaryImpl = MakeHolder<TMMapUnigramDictionaryImpl>(ptr);
    } else {
        switch (gramOrder) {
            case 2:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<2>>(ptr);
                break;
            case 3:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<3>>(ptr);
                break;
            case 4:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<4>>(ptr);
                break;
            case 5:
                DictionaryImpl = MakeHolder<TMMapMultigramDictionaryImpl<5>>(ptr);
                break;
            default:
                Y_ENSURE(false, "Unsupported gram order: " << gramOrder << ".");
        }
    }

    ptr += dictionaryMetaInfoBufferSize;
    const ui64 restSize = size - (ptr - reinterpret_cast<const ui8*>(data));
    Y_ENSURE(restSize + 16 + dictionaryMetaInfoBufferSize == totalSize, "Incorrect data");
    DictionaryImpl->InitFromMemory(ptr, restSize);
}

size_t TMMapDictionary::CalculateExpectedSize(const void *data, size_t size) {
    const ui8* ptr = reinterpret_cast<const ui8*>(data);
    Y_ENSURE(size >= 16 + 8); // проверяем, что можем прочитать total size и заголовок
    Y_ENSURE(!std::memcmp(ptr, MAGIC, MAGIC_SIZE));
    ptr += 16;
    const ui64 totalSize = *reinterpret_cast<const ui64*>(ptr);
    Y_ENSURE(totalSize + 16 <= size);
    return totalSize + 16;
}
