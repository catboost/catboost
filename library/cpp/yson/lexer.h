#pragma once

#include "public.h"
#include "token.h"

#include <util/generic/ptr.h>

namespace NYT {
    ////////////////////////////////////////////////////////////////////////////////

    class TStatelessLexer {
    public:
        TStatelessLexer();

        ~TStatelessLexer();

        size_t GetToken(const TStringBuf& data, TToken* token);

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };

    ////////////////////////////////////////////////////////////////////////////////

}
