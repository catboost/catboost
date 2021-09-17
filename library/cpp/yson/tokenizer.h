#pragma once

#include "public.h"
#include "lexer.h"

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    class TTokenizer {
    public:
        explicit TTokenizer(const TStringBuf& input);

        bool ParseNext();
        const TToken& CurrentToken() const;
        ETokenType GetCurrentType() const;
        TStringBuf GetCurrentSuffix() const;
        const TStringBuf& CurrentInput() const;

    private:
        TStringBuf Input;
        TToken Token;
        TStatelessLexer Lexer;
        size_t Parsed;
    };

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
