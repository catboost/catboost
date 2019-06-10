#include "text_column_builder.h"

using namespace NCB;

NCB::TText NCB::TokensToText(
    const IDictionary& dictionary,
    TConstArrayRef<TStringBuf> tokens) {

    TText result;
    TVector<ui32> tokenIds;
    dictionary.Apply(tokens, &tokenIds);
    for (const auto& tokenId : tokenIds) {
        result[TTokenId(tokenId)]++;
    }
    return result;
}

void TTextColumnBuilder::AddText(ui32 index, const TStringBuf text) {
    CB_ENSURE_INTERNAL(index < Texts.size(), "Text index is out of range");
    TVector<TStringBuf> tokens;
    Tokenizer->Tokenize(text, &tokens);
    Texts[index] = TokensToText(*Dictionary, tokens);
}

TTextColumn TTextColumnBuilder::Build() {
    CB_ENSURE_INTERNAL(!WasBuilt, "Build could be done only once");
    WasBuilt = true;
    return TMaybeOwningConstArrayHolder<TText>::CreateOwning(std::move(Texts));
}
