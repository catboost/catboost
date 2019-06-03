#include "tokenizer.h"
#include <catboost/libs/helpers/exception.h>
#include <util/generic/string.h>
#include <util/string/split.h>

using namespace NCB;


namespace {
    class TNaiveTokenizer : public NCB::ITokenizer {
    public:
        void Tokenize(TStringBuf inputString, TVector<TStringBuf>* tokens) const override {
            tokens->clear();
            for (const auto& token : StringSplitter(inputString).Split(' ').SkipEmpty()) {
                tokens->push_back(token);
            }
        }
    };
}

NCB::TTokenizerPtr NCB::CreateTokenizer(ETokenizerType tokenizerType) {
    if (tokenizerType == ETokenizerType::Naive)
        return new TNaiveTokenizer();
    else {
        CB_ENSURE(false, "Currently supported only naive tokenizer");
    }
}
