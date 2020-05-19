#include "split.h"

#include <library/cpp/unittest/registar.h>

#include <util/charset/wide.h>

static TVector<TUtf16String> ConvertFromUTF8(const TVector<TString>& utf8text) {
    TVector<TUtf16String> wideText;
    for (const auto& w : utf8text) {
        wideText.push_back(UTF8ToWide(w));
    }
    return wideText;
}

Y_UNIT_TEST_SUITE(TTokenizerSplitTest) {
    Y_UNIT_TEST(TestSplitIntoSentences) {
        TUtf16String text = u"— Поднимите меня! — попросила Раневская. — Народные артистки на дороге не валяются...";

        TVector<TString> utf8sents;
        utf8sents.push_back("— Поднимите меня! — попросила Раневская. ");
        utf8sents.push_back("— Народные артистки на дороге не валяются...");

        UNIT_ASSERT_EQUAL(SplitIntoSentences(text), ConvertFromUTF8(utf8sents));
    }

    Y_UNIT_TEST(TestSplitIntoWords) {
        TUtf16String text = u"Первый ход — Е2-Е4, а там... А там посмотрим.";

        TVector<TString> utf8w;
        utf8w.push_back("Первый");
        utf8w.push_back("ход");
        utf8w.push_back("а");
        utf8w.push_back("там");
        utf8w.push_back("А");
        utf8w.push_back("там");
        utf8w.push_back("посмотрим");

        UNIT_ASSERT_EQUAL(SplitIntoTokens(text), ConvertFromUTF8(utf8w));
    }

    Y_UNIT_TEST(TestSplitIntoNonPunct) {
        TUtf16String text = u"Первый ход — Е2-Е4, а там... А там посмотрим.";

        TVector<TString> utf8w;
        utf8w.push_back("Первый");
        utf8w.push_back("ход");
        utf8w.push_back("Е2");
        utf8w.push_back("Е4");
        utf8w.push_back("а");
        utf8w.push_back("там");
        utf8w.push_back("А");
        utf8w.push_back("там");
        utf8w.push_back("посмотрим");

        auto&& params = TTokenizerSplitParams(TTokenizerSplitParams::NOT_PUNCT);
        UNIT_ASSERT_EQUAL(SplitIntoTokens(text, params), ConvertFromUTF8(utf8w));
    }

    Y_UNIT_TEST(TestSplitIntoAll) {
        TUtf16String text = u"Первый ход — Е2-Е4, а там... А там посмотрим.";

        TVector<TString> utf8w;
        utf8w.push_back("Первый");
        utf8w.push_back(" ");
        utf8w.push_back("ход");
        utf8w.push_back(" — ");
        utf8w.push_back("Е2");
        utf8w.push_back("-");
        utf8w.push_back("Е4");
        utf8w.push_back(", ");
        utf8w.push_back("а");
        utf8w.push_back(" ");
        utf8w.push_back("там");
        utf8w.push_back("... ");
        utf8w.push_back("А");
        utf8w.push_back(" ");
        utf8w.push_back("там");
        utf8w.push_back(" ");
        utf8w.push_back("посмотрим");
        utf8w.push_back(".");

        const TTokenizerSplitParams::THandledMask ALL(NLP_WORD,
            NLP_INTEGER, NLP_FLOAT, NLP_MARK, NLP_MISCTEXT, NLP_SENTBREAK);

        auto&& params = TTokenizerSplitParams(ALL);
        UNIT_ASSERT_EQUAL(SplitIntoTokens(text, params), ConvertFromUTF8(utf8w));
    }
}
