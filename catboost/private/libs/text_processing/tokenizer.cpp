#include "tokenizer.h"

void NCB::TTokenizer::Tokenize(TStringBuf inputString, TTokensWithBuffer* tokens) {
    if (TokenizerImpl.NeedToModifyTokens()) {
        TokenizerImpl.Tokenize(inputString, &tokens->Data);
        tokens->View.clear();
        for (const auto& token: tokens->Data) {
            tokens->View.emplace_back(token);
        }
    } else {
        TokenizerImpl.TokenizeWithoutCopy(inputString, &tokens->View);
    }
}

NCB::TTokenizerPtr NCB::CreateTokenizer() {
    return new TTokenizer();
}
