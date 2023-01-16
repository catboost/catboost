#include "lexer.h"
#include "lexer_detail.h"
#include "token.h"

#include <util/generic/ptr.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    class TStatelessLexer::TImpl {
    private:
        THolder<TStatelessYsonLexerImplBase> Impl;

    public:
        TImpl(bool enableLinePositionInfo = false)
            : Impl(enableLinePositionInfo
                       ? static_cast<TStatelessYsonLexerImplBase*>(new TStatelesYsonLexerImpl<true>())
                       : static_cast<TStatelessYsonLexerImplBase*>(new TStatelesYsonLexerImpl<false>()))
                   {
        }

        size_t GetToken(const TStringBuf& data, TToken* token) {
            return Impl->GetToken(data, token);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    TStatelessLexer::TStatelessLexer()
        : Impl(new TImpl())
    {
    }

    TStatelessLexer::~TStatelessLexer() {
    }

    size_t TStatelessLexer::GetToken(const TStringBuf& data, TToken* token) {
        return Impl->GetToken(data, token);
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
