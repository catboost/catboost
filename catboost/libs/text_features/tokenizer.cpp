#include "tokenizer.h"
#include <util/generic/string.h>
#include <util/string/split.h>

using namespace NCB;

namespace {

    class TNaiveTokenizer : public ITokenizer {
    public:
        void Tokenize(TStringBuf inputString, TVector<TString>* tokens) const override {
            tokens->clear();
            for (const auto& token : StringSplitter(inputString).Split(' ')) {
                tokens->push_back(TString(token));
            }
        }
    };
}

TTokenizerPtr CreateTokenizer() {
    return new TNaiveTokenizer;
}
