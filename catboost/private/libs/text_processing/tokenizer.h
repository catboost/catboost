#pragma once

#include <catboost/libs/helpers/guid.h>

#include <library/cpp/text_processing/tokenizer/tokenizer.h>

namespace NCB {

    struct TTokensWithBuffer {
        TVector<TStringBuf> View;
        TVector<TString> Data;
    };

    class TTokenizer : public TThrRefBase {
    public:
        TTokenizer() = default;
        explicit TTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options);

        TGuid Id() const;
        NTextProcessing::NTokenizer::TTokenizerOptions Options() const;
        void Tokenize(TStringBuf inputString, TTokensWithBuffer* tokens);

        void Save(IOutputStream* stream) const;
        void Load(IInputStream* stream);
    private:
        TGuid Guid;
        NTextProcessing::NTokenizer::TTokenizer TokenizerImpl;

        static constexpr std::array<char, 12> TokenizerMagic = {"TokenizerV1"};
        static constexpr ui32 MagicSize = TokenizerMagic.size();
        static constexpr ui32 Alignment = 16;
    };

    using TTokenizerPtr = TIntrusivePtr<TTokenizer>;

    TTokenizerPtr CreateTokenizer(const NTextProcessing::NTokenizer::TTokenizerOptions& options = {});
}


