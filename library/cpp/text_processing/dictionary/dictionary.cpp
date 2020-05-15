#include "dictionary.h"

#include "frequency_based_dictionary.h"

using NTextProcessing::NDictionary::IDictionary;
using NTextProcessing::NDictionary::TDictionary;
using NTextProcessing::NDictionary::TTokenId;

TIntrusivePtr<IDictionary> IDictionary::Load(IInputStream *stream) {
    auto dictionary = MakeIntrusive<TDictionary>();
    dictionary->Load(stream);
    return dictionary;
}

void IDictionary::GetTokens(TConstArrayRef<TTokenId> tokenIds, TVector<TString>* tokens) const {
    tokens->clear();
    tokens->reserve(tokenIds.size());
    for (auto tokenId : tokenIds) {
        tokens->emplace_back(GetToken(tokenId));
    }
}
