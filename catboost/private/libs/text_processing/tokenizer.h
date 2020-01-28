#pragma once

#include <catboost/libs/helpers/guid.h>

#include <library/text_processing/tokenizer/tokenizer.h>

namespace NCB {

    struct TTokensWithBuffer {
        TVector<TStringBuf> View;
        TVector<TString> Data;
    };

    class TTokenizer : public TThrRefBase {
    public:
        explicit TTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options);
        TGuid Id() const;
        void Tokenize(TStringBuf inputString, TTokensWithBuffer* tokens);
    private:
        TGuid Guid;
        NTextProcessing::NTokenizer::TTokenizer TokenizerImpl;
    };

    using TTokenizerPtr = TIntrusivePtr<TTokenizer>;

    TTokenizerPtr CreateTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options = {});
}


