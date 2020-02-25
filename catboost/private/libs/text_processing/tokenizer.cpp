#include "tokenizer.h"

#include <catboost/libs/helpers/serialization.h>

NCB::TTokenizer::TTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options)
    : Guid(CreateGuid())
    , TokenizerImpl(options)
{
}

NCB::TGuid NCB::TTokenizer::Id() const {
    return Guid;
}

NTextProcessing::NTokenizer::TTokenizerOptions NCB::TTokenizer::Options() const {
    return TokenizerImpl.GetOptions();
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

void NCB::TTokenizer::Save(IOutputStream *stream) const {
    WriteMagic(TokenizerMagic.data(), MagicSize, Alignment, stream);
    Guid.Save(stream);
    TokenizerImpl.Save(stream);
}

void NCB::TTokenizer::Load(IInputStream *stream) {
    ReadMagic(TokenizerMagic.data(), MagicSize, Alignment, stream);
    Guid.Load(stream);
    TokenizerImpl.Load(stream);
}

NCB::TTokenizerPtr NCB::CreateTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options) {
    return new TTokenizer(options);
}
