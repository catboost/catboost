#include "text_column_builder.h"

using namespace NCB;


void TTextColumnBuilder::AddText(ui32 index, const TStringBuf text) {
    CB_ENSURE_INTERNAL(index < Texts.size(), "Text index is out of range");
    TTokensWithBuffer tokens;
    Tokenizer->Tokenize(text, &tokens);
    Dictionary->Apply(tokens.View, &Texts[index]);
}

TVector<TText> TTextColumnBuilder::Build() {
    CB_ENSURE_INTERNAL(!WasBuilt, "Build could be done only once");
    WasBuilt = true;
    return Texts;
}
