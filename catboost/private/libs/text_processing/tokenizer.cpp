#include "tokenizer.h"

NCB::TTokenizer::TTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options)
    : TokenizerImpl(options)
{
}

NCB::TGuid NCB::TTokenizer::Id() const {
    return Guid;
}

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

NCB::TTokenizerPtr NCB::CreateTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options) {
    return new TTokenizer(options);
}
