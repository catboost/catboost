#pragma once

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>


namespace NCB {

    //TODO(noxoomo, nikitxskv): move to library after text-processing tokenizer will be available
    class ITokenizer : public TThrRefBase {
    public:
        virtual void Tokenize(TStringBuf inputString, TVector<TString>* tokens) const = 0;
    };

    using TTokenizerPtr = TIntrusivePtr<ITokenizer>;

    TTokenizerPtr CreateTokenizer();

}


