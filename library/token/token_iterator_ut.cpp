#include <library/charset/wide.h>
#include <library/cpp/unittest/registar.h>
#include <library/cpp/tokenizer/tokenizer.h>
#include <kernel/qtree/request/request.h>
#include <kernel/qtree/request/req_node.h>
#include "token_iterator.h"

namespace {
    char GetNlpTypeChar(NLP_TYPE type) {
        switch (type) {
            case NLP_WORD:
                return 'W';
            case NLP_MARK:
                return 'M';
            case NLP_FLOAT:
                return 'F';
            case NLP_INTEGER:
                return 'I';
            default:
                return '-';
        }
    }

    class TTokenHandler: public ITokenHandler {
        TUtf16String Text;
        TString Types;

    public:
        void OnToken(const TWideToken& token, size_t /*origleng*/, NLP_TYPE type) override {
            if (type != NLP_WORD && type != NLP_INTEGER && type != NLP_FLOAT && type != NLP_MARK)
                return;
            if (!Text.empty())
                Text += wchar16(' ');
            Text += TUtf16String(token.Token, token.Leng);
            Types += GetNlpTypeChar(type);
        }
        TString GetText() const {
            return WideToUTF8(Text);
        }
        TString GetTypes() const {
            return Types;
        }
    };

}

class TTokenIteratorTest: public TTestBase {
    UNIT_TEST_SUITE(TTokenIteratorTest);
    UNIT_TEST(Test);
    UNIT_TEST_SUITE_END();

public:
    void Test();

private:
    void TestCase(const char* s1, const char* s2, const char* t2, const char* s3 = nullptr);
    void WalkTree(const TRequestNode& node, TUtf16String& w1, TString& t1);
    void AddWords(const TWideToken& tok, TUtf16String& w1, TString& t1);
};

UNIT_TEST_SUITE_REGISTRATION(TTokenIteratorTest);

void TTokenIteratorTest::Test() {
    TestCase("v1.0", "v 1.0", "WF"); // floats have priority
    TestCase("1.2b", "1.2 b", "FW");

    TestCase("1a-b", "1a b", "MW"); // marks have priority
    TestCase("a-b2", "a b2", "WM");

    TestCase("a-b-c-d/e'd'f'g/h-i'j", "a-b-c-d e'd'f'g h-i'j", "WWW");
    TestCase("v1.x_3@a-b1z45/c+d", "v1 x 3 a b1z45 c d", "MWIWMWW");

    TestCase("1.2.3.4.5", "1.2 3.4 5", "FFI");
    TestCase("v1.0.1.5", "v 1.0 1.5", "WFF");
    TestCase("1.0.1d", "1.0 1d", "FM");
    TestCase("1.0.1.5d", "1.0 1.5 d", "FFW");
    TestCase("1.0.1.5.7d", "1.0 1.5 7d", "FFM");

    TestCase("1S7.7", "1S 7.7", "MF");

    TestCase("a+.b#.c-d", "a+ b# c-d", "WWW", "a+ b# c-d");
    TestCase("a-b-c+.d-e", "a-b-c+ d-e", "WW", "a-b-c+ d-e");
    TestCase("a-b+c-d++e-f+", "a-b c-d+ e-f+", "WWW", "a-b c-d+ e-f+");
    TestCase("a+b++", "a b++", "WW", "a b++");
    TestCase("a++b+", "a+ b+", "WW", "a+ b+");

    TestCase("1+.7#.3.2", "1+ 7# 3.2", "IIF", "1 7 3.2");
    TestCase("1.2.3+.4.5", "1.2 3+ 4.5", "FIF", "1.2 3 4.5");
    TestCase("1.2.3.4+5.6", "1.2 3.4 5.6", "FFF", "1.2 3.4 5.6");

    TestCase("a1+b2++", "a1 b2++", "MM", "a1 b2");
    TestCase("a1++b2+", "a1+ b2+", "MM", "a1 b2");
    TestCase("1a+2b++", "1a 2b++", "MM", "1a 2b");
    TestCase("1a++2b+", "1a+ 2b+", "MM", "1a 2b");
    TestCase("1.2+3.4++", "1.2 3.4++", "FF", "1.2 3.4");
    TestCase("1.2++3.4+", "1.2+ 3.4+", "FF", "1.2 3.4");

    TestCase("0.5.5b+", "0.5 5b+", "FM", "0.5 5b");

    TestCase("-mail:turanova@kspu.ru", "mail turanova kspu ru", "WWWW");
    TestCase("\"%one +!two\"", "one two", "WW");

    const wchar16 text1[] = {0xDB00, 'a', 0xDB00, 0xDB00, 'b', 0xDB00, 'c', 0xDB00, 0}; // surrogate leads
    const wchar16 result1[] = {0xFFFD, ' ', 'a', ' ', 0xFFFD, ' ', 0xFFFD, ' ', 'b', ' ', 0xFFFD, ' ', 'c', ' ', 0xFFFD, 0};
    TestCase(WideToUTF8(text1).c_str(), WideToUTF8(result1).c_str(), "WWWWWWWW");
    const wchar16 text2[] = {'x', 0xDC00, ' ', 0xDC00, 'y', 0xDC00, 'z', 0xDC00, 0}; // surrogate tails
    const wchar16 result2[] = {'x', ' ', 0xFFFD, ' ', 0xFFFD, ' ', 'y', ' ', 0xFFFD, ' ', 'z', ' ', 0xFFFD, 0};
    TestCase(WideToUTF8(text2).c_str(), WideToUTF8(result2).c_str(), "WWWWWWW");
}

//! @param s2    expected result for TTokenIteartor::Get()
//! @param s3    expected result for TTokenIteartor::GetMultitoken() (it used in TNlpParser::MakeEntry())
void TTokenIteratorTest::TestCase(const char* s1, const char* s2, const char* t2, const char* s3) {
    {
        tRequest req;
        const tRequest::TNodePtr p(req.Parse(UTF8ToWide(s1)));
        UNIT_ASSERT(p.Get());
        TUtf16String w1;
        TString t1;
        WalkTree(*p, w1, t1);
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(w1).c_str(), s2);
        UNIT_ASSERT_STRINGS_EQUAL(t1.c_str(), t2);
    }
    if (s3) {
        TTokenHandler handler;
        TNlpTokenizer tokenizer(handler, true); // (backwardCompatible == true) -> use TTokenIterator in TNlpParser::MakeEntry()
        tokenizer.Tokenize(s1, strlen(s1));
        UNIT_ASSERT_STRINGS_EQUAL(handler.GetText().c_str(), s3);
        UNIT_ASSERT_STRINGS_EQUAL(handler.GetTypes().c_str(), t2);
    }
}

void TTokenIteratorTest::WalkTree(const TRequestNode& node, TUtf16String& w1, TString& t1) {
    if (IsWordOrMultitoken(node)) {
        UNIT_ASSERT(!node.Left && !node.Right);
        AddWords(node.GetMultitoken(), w1, t1);
        return;
    }
    if (node.Left)
        WalkTree(*node.Left, w1, t1);
    if (node.Right)
        WalkTree(*node.Right, w1, t1);
}

void TTokenIteratorTest::AddWords(const TWideToken& tok, TUtf16String& w1, TString& t1) {
    TTokenIterator it(tok);
    while (it.Next()) {
        if (!w1.empty())
            w1.append(' ');
        t1.append(GetNlpTypeChar(it.GetNlpType()));
        const TTokenStructure& tokens = it.Get();
        for (size_t i = 0; i < tokens.size(); ++i) {
            const TCharSpan& s = tokens[i];
            const size_t len = s.Len + s.SuffixLen;
            w1.append(tok.Token + s.Pos, len);
            if (i < tokens.size() - 1 && s.TokenDelim != TOKDELIM_NULL) {
                const size_t pos = s.Pos + len;
                w1.append(tok.Token + pos, tokens[i + 1].Pos - pos);
            }
        }
    }
}
