#pragma once

#include "lemmer_impl.h"
#include "options.h"

#include <util/generic/ptr.h>

namespace NTextProcessing::NTokenizer {

    class TTokenizer : public TMoveOnly {
    public:
        TTokenizer();
        explicit TTokenizer(const TTokenizerOptions& options);

        void Tokenize(
            TStringBuf inputString,
            TVector<TString>* tokens,
            TVector<ETokenType>* tokenTypes = nullptr
        ) const;
        TVector<TString> Tokenize(TStringBuf inputString) const;

        void TokenizeWithoutCopy(TStringBuf inputString, TVector<TStringBuf>* tokens) const;
        TVector<TStringBuf> TokenizeWithoutCopy(TStringBuf inputString) const;

        TTokenizerOptions GetOptions() const;

        bool NeedToModifyTokens() const;

        void Save(IOutputStream* stream) const;
        void Load(IInputStream* stream);

        bool operator==(const TTokenizer& rhs) const;
        bool operator!=(const TTokenizer& rhs) const;

    private:
        void Initialize();

        TTokenizerOptions Options;
        THolder<ILemmerImplementation> Lemmer;
        bool NeedToModifyTokensFlag = false;
    };

}
