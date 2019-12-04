#pragma once

#include <library/text_processing/tokenizer/tokenizer.h>

namespace NCB {

    class TTokenizer : public TThrRefBase {
    public:
        void Tokenize(TStringBuf inputString, TVector<TStringBuf>* tokens);
    private:
        NTextProcessing::NTokenizer::TTokenizer TokenizerImpl;
    };

    using TTokenizerPtr = TIntrusivePtr<TTokenizer>;

    TTokenizerPtr CreateTokenizer();
}


