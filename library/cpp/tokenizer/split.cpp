#include "split.h"

#include "tokenizer.h"

namespace {
    class TSimpleTokenHandler: public ITokenHandler {
    public:
        TSimpleTokenHandler(TVector<TUtf16String>* outTokens, const TTokenizerSplitParams& params)
            : Tokens(outTokens)
            , Params(params)
        {
        }

        void OnToken(const TWideToken& token, size_t, NLP_TYPE type) override {
            if (!Params.HandledMask.SafeTest(type)) {
                return;
            }

            Tokens->push_back(TUtf16String(token.Token, token.Leng));
        }

    private:
        TVector<TUtf16String>* Tokens;
        TTokenizerSplitParams Params;
    };

    class TSimpleSentenceHandler: public ITokenHandler {
    public:
        TSimpleSentenceHandler(TVector<TUtf16String>* sentences)
            : Sentences(sentences)
        {
        }

        void OnToken(const TWideToken& token, size_t, NLP_TYPE type) override {
            CurToken += token.Text();

            if (type == NLP_SENTBREAK || type == NLP_PARABREAK) {
                Flush();
            }
        }

        void Flush() {
            if (!CurToken.empty()) {
                Sentences->push_back(CurToken);

                CurToken = TUtf16String();
            }
        }

    private:
        TUtf16String CurToken;
        TVector<TUtf16String>* Sentences;
    };
}

const TTokenizerSplitParams::THandledMask TTokenizerSplitParams::WORDS(NLP_WORD);
const TTokenizerSplitParams::THandledMask TTokenizerSplitParams::NOT_PUNCT(NLP_WORD, NLP_INTEGER, NLP_FLOAT, NLP_MARK);

TVector<TUtf16String> SplitIntoTokens(const TUtf16String& text, const TTokenizerSplitParams& params) {
    TVector<TUtf16String> words;

    TSimpleTokenHandler handler(&words, params);
    TNlpTokenizer tokenizer(handler, params.BackwardCompatibility);
    TTokenizerOptions opts { params.SpacePreserve, params.TokenizerLangMask, params.UrlDecode };
    tokenizer.Tokenize(text.data(), text.size(), opts);

    return words;
}

TVector<TUtf16String> SplitIntoSentences(const TUtf16String& text, const TTokenizerSplitParams& params) {
    TVector<TUtf16String> sentences;

    TSimpleSentenceHandler handler(&sentences);
    TNlpTokenizer tokenizer(handler, params.BackwardCompatibility);
    TTokenizerOptions opts { params.SpacePreserve, params.TokenizerLangMask, params.UrlDecode };
    tokenizer.Tokenize(text.data(), text.size(), opts);
    handler.Flush();

    return sentences;
}
