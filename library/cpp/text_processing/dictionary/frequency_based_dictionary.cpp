#include "frequency_based_dictionary.h"
#include "frequency_based_dictionary_impl.h"
#include "util.h"

#include <library/cpp/json/json_reader.h>

#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>

using namespace NTextProcessing::NDictionary;

TDictionary::TDictionary() = default;
TDictionary::TDictionary(TDictionary&&) = default;
TDictionary::~TDictionary() = default;

TDictionary::TDictionary(THolder<IDictionaryImpl> dictionaryImpl)
    : DictionaryImpl(std::move(dictionaryImpl))
{
}

TTokenId TDictionary::Apply(TStringBuf token) const {
    return DictionaryImpl->Apply(token);
}

void TDictionary::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    DictionaryImpl->Apply(tokens, tokenIds, unknownTokenPolicy);
}

void TDictionary::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokenIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    DictionaryImpl->Apply(tokens, tokenIds, unknownTokenPolicy);
}

ui32 TDictionary::Size() const {
    return DictionaryImpl->Size();
}

TString TDictionary::GetToken(TTokenId tokenId) const {
    return DictionaryImpl->GetToken(tokenId);
}

ui64 TDictionary::GetCount(TTokenId tokenId) const {
    return DictionaryImpl->GetCount(tokenId);
}

TVector<TString> TDictionary::GetTopTokens(ui32 topSize) const {
    return DictionaryImpl->GetTopTokens(topSize);
}

void TDictionary::ClearStatsData() {
    return DictionaryImpl->ClearStatsData();
}

TTokenId TDictionary::GetUnknownTokenId() const {
    return DictionaryImpl->GetUnknownTokenId();
}

TTokenId TDictionary::GetEndOfSentenceTokenId() const {
    return DictionaryImpl->GetEndOfSentenceTokenId();
}

TTokenId TDictionary::GetMinUnusedTokenId() const {
    return DictionaryImpl->GetMinUnusedTokenId();
}

const TDictionaryOptions& TDictionary::GetDictionaryOptionsRef() const {
    return DictionaryImpl->GetDictionaryOptionsRef();
}

void TDictionary::Save(IOutputStream* stream) const {
    DictionaryImpl->Save(stream);
}

void TDictionary::Load(IInputStream* stream) {
    NJson::TJsonValue optionsJson;
    NJson::ReadJsonTree(stream->ReadLine(), &optionsJson);
    auto dictionaryOptions = JsonToDictionaryOptions(optionsJson);
    const bool isNewFormat = optionsJson.Has(DICT_FORMAT_KEY) && optionsJson[DICT_FORMAT_KEY].GetString() == DICT_NEW_FORMAT_DESC;
    if (dictionaryOptions.TokenLevelType == ETokenLevelType::Letter || dictionaryOptions.GramOrder == 1) {
        DictionaryImpl = MakeHolder<TUnigramDictionaryImpl>(dictionaryOptions);
    } else {
        switch (dictionaryOptions.GramOrder) {
            case 2:
                DictionaryImpl = MakeHolder<TMultigramDictionaryImpl<2>>(dictionaryOptions);
                break;
            case 3:
                DictionaryImpl = MakeHolder<TMultigramDictionaryImpl<3>>(dictionaryOptions);
                break;
            case 4:
                DictionaryImpl = MakeHolder<TMultigramDictionaryImpl<4>>(dictionaryOptions);
                break;
            case 5:
                DictionaryImpl = MakeHolder<TMultigramDictionaryImpl<5>>(dictionaryOptions);
                break;
            default:
                Y_ENSURE(false, "Unsupported gram order: " << dictionaryOptions.GramOrder << ".");
        }
    }

    DictionaryImpl->Load(stream, isNewFormat);
}
