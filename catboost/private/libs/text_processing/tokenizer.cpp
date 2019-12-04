#include "tokenizer.h"

void NCB::TTokenizer::Tokenize(TStringBuf inputString, TVector<TStringBuf>* tokens) {
    TokenizerImpl.TokenizeWithoutCopy(inputString, tokens);
}

NCB::TTokenizerPtr NCB::CreateTokenizer() {
    return new TTokenizer();
}
