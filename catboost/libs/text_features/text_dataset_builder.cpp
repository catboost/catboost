#include "text_dataset_builder.h"

using namespace NCB;

NCB::TText NCB::TokensToText(
    const IDictionary& dictionary,
    TConstArrayRef<TString> tokens) {

    TText result;
    TVector<ui32> tokenIds;
    dictionary.Apply(tokens, &tokenIds);
    for (const auto& tokenId : tokenIds) {
        result[TTokenId(tokenId)]++;
    }
    return result;
}

void TTextDataSetBuilder::AddText(const TStringBuf text) {
    TVector<TString> tokens;
    Tokenizer->Tokenize(text, &tokens);
    Texts.push_back(TokensToText(*Dictionary, tokens));
}

TIntrusivePtr<TTextDataSet> TTextDataSetBuilder::Build() {
    CB_ENSURE(!WasBuilt, "Build could be done only once");
    WasBuilt = true;
    return new NCB::TTextDataSet(std::move(Texts));
}
