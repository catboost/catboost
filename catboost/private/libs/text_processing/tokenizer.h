#pragma once

#include <library/text_processing/tokenizer/tokenizer.h>

namespace NCB {

    struct TTokensWithBuffer {
        TVector<TStringBuf> View;
        TVector<TString> Data;
    };

    class TTokenizer : public TThrRefBase {
    public:
        void Tokenize(TStringBuf inputString, TTokensWithBuffer* tokens);
    private:
        NTextProcessing::NTokenizer::TTokenizer TokenizerImpl;
    };

    using TTokenizerPtr = TIntrusivePtr<TTokenizer>;

    TTokenizerPtr CreateTokenizer();
}


