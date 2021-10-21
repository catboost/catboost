#include "tokenizer.h"

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    TTokenizer::TTokenizer(const TStringBuf& input)
        : Input(input)
        , Parsed(0)
    {
    }

    bool TTokenizer::ParseNext() {
        Input = Input.Tail(Parsed);
        Token.Reset();
        Parsed = Lexer.GetToken(Input, &Token);
        return !CurrentToken().IsEmpty();
    }

    const TToken& TTokenizer::CurrentToken() const {
        return Token;
    }

    ETokenType TTokenizer::GetCurrentType() const {
        return CurrentToken().GetType();
    }

    TStringBuf TTokenizer::GetCurrentSuffix() const {
        return Input.Tail(Parsed);
    }

    const TStringBuf& TTokenizer::CurrentInput() const {
        return Input;
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
