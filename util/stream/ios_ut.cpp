#include "output.h"
#include "tokenizer.h"
#include "buffer.h"
#include "buffered.h"
#include "walk.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/cast.h>
#include <util/memory/tempbuf.h>
#include <util/charset/wide.h>

#include <string>

class TStreamsTest: public TTestBase {
    UNIT_TEST_SUITE(TStreamsTest);
    UNIT_TEST(TestGenericRead);
    UNIT_TEST(TestGenericWrite);
    UNIT_TEST(TestReadLine);
    UNIT_TEST(TestMemoryStream);
    UNIT_TEST(TestBufferedIO);
    UNIT_TEST(TestBufferStream);
    UNIT_TEST(TestStringStream);
    UNIT_TEST(TestWtrokaInput);
    UNIT_TEST(TestStrokaInput);
    UNIT_TEST(TestReadTo);
    UNIT_TEST(TestWtrokaOutput);
    UNIT_TEST(TestIStreamOperators);
    UNIT_TEST(TestWchar16Output);
    UNIT_TEST(TestWchar32Output);
    UNIT_TEST(TestUtf16StingOutputByChars);
    UNIT_TEST_SUITE_END();

public:
    void TestGenericRead();
    void TestGenericWrite();
    void TestReadLine();
    void TestMemoryStream();
    void TestBufferedIO();
    void TestBufferStream();
    void TestStringStream();
    void TestWtrokaInput();
    void TestStrokaInput();
    void TestWtrokaOutput();
    void TestIStreamOperators();
    void TestReadTo();
    void TestWchar16Output();
    void TestWchar32Output();
    void TestUtf16StingOutputByChars();
};

UNIT_TEST_SUITE_REGISTRATION(TStreamsTest);

void TStreamsTest::TestIStreamOperators() {
    TString data("first line\r\nsecond\t\xd1\x82\xd0\xb5\xd1\x81\xd1\x82 line\r\n 1 -4 59 4320000000009999999 c\n -1.5 1e-110");
    TStringInput si(data);

    TString l1;
    TString l2;
    TString l3;
    TUtf16String w1;
    TString l4;
    ui16 i1;
    i16 i2;
    i32 i3;
    ui64 i4;
    char c1;
    unsigned char c2;
    float f1;
    double f2;

    si >> l1 >> l2 >> l3 >> w1 >> l4 >> i1 >> i2 >> i3 >> i4 >> c1 >> c2 >> f1 >> f2;

    UNIT_ASSERT_EQUAL(l1, "first");
    UNIT_ASSERT_EQUAL(l2, "line");
    UNIT_ASSERT_EQUAL(l3, "second");
    UNIT_ASSERT_EQUAL(l4, "line");
    UNIT_ASSERT_EQUAL(i1, 1);
    UNIT_ASSERT_EQUAL(i2, -4);
    UNIT_ASSERT_EQUAL(i3, 59);
    UNIT_ASSERT_EQUAL(i4, 4320000000009999999ULL);
    UNIT_ASSERT_EQUAL(c1, 'c');
    UNIT_ASSERT_EQUAL(c2, '\n');
    UNIT_ASSERT_EQUAL(f1, -1.5);
    UNIT_ASSERT_EQUAL(f2, 1e-110);
}

void TStreamsTest::TestStringStream() {
    TStringStream s;

    s << "qw\r\n1234"
      << "\n"
      << 34;

    UNIT_ASSERT_EQUAL(s.ReadLine(), "qw");
    UNIT_ASSERT_EQUAL(s.ReadLine(), "1234");

    s << "\r\n"
      << 123.1;

    UNIT_ASSERT_EQUAL(s.ReadLine(), "34");
    UNIT_ASSERT_EQUAL(s.ReadLine(), "123.1");

    UNIT_ASSERT_EQUAL(s.Str(), "qw\r\n1234\n34\r\n123.1");

    // Test stream copying
    TStringStream sc = s;

    s << "-666-" << 13;
    sc << "-777-" << 0 << "JackPot";

    UNIT_ASSERT_EQUAL(s.Str(), "qw\r\n1234\n34\r\n123.1-666-13");
    UNIT_ASSERT_EQUAL(sc.Str(), "qw\r\n1234\n34\r\n123.1-777-0JackPot");

    TStringStream ss;
    ss = s;
    s << "... and some trash";
    UNIT_ASSERT_EQUAL(ss.Str(), "qw\r\n1234\n34\r\n123.1-666-13");
}

void TStreamsTest::TestGenericRead() {
    TString s("1234567890");
    TStringInput si(s);
    char buf[1024];

    UNIT_ASSERT_EQUAL(si.Read(buf, 6), 6);
    UNIT_ASSERT_EQUAL(memcmp(buf, "123456", 6), 0);
    UNIT_ASSERT_EQUAL(si.Read(buf, 6), 4);
    UNIT_ASSERT_EQUAL(memcmp(buf, "7890", 4), 0);
}

void TStreamsTest::TestGenericWrite() {
    TString s;
    TStringOutput so(s);

    so.Write("123456", 6);
    so.Write("7890", 4);

    UNIT_ASSERT_EQUAL(s, "1234567890");
}

void TStreamsTest::TestReadLine() {
    TString data("1234\r\n5678\nqw");
    TStringInput si(data);

    UNIT_ASSERT_EQUAL(si.ReadLine(), "1234");
    UNIT_ASSERT_EQUAL(si.ReadLine(), "5678");
    UNIT_ASSERT_EQUAL(si.ReadLine(), "qw");
}

void TStreamsTest::TestMemoryStream() {
    char buf[1024];
    TMemoryOutput mo(buf, sizeof(buf));
    bool ehandled = false;

    try {
        for (size_t i = 0; i < sizeof(buf) + 1; ++i) {
            mo.Write(i % 127);
        }
    } catch (...) {
        ehandled = true;
    }

    UNIT_ASSERT_EQUAL(ehandled, true);

    for (size_t i = 0; i < sizeof(buf); ++i) {
        UNIT_ASSERT_EQUAL(buf[i], (char)(i % 127));
    }
}

class TMyStringOutput: public IOutputStream {
public:
    inline TMyStringOutput(TString& s, size_t buflen) noexcept
        : S_(s)
        , BufLen_(buflen)
    {
    }

    ~TMyStringOutput() override = default;

    void DoWrite(const void* data, size_t len) override {
        S_.Write(data, len);
        UNIT_ASSERT(len < BufLen_ || ((len % BufLen_) == 0));
    }

    void DoWriteV(const TPart* p, size_t count) override {
        TString s;

        for (size_t i = 0; i < count; ++i) {
            s.append((const char*)p[i].buf, p[i].len);
        }

        DoWrite(s.data(), s.size());
    }

private:
    TStringOutput S_;
    const size_t BufLen_;
};

void TStreamsTest::TestBufferedIO() {
    TString s;

    {
        const size_t buflen = 7;
        TBuffered<TMyStringOutput> bo(buflen, s, buflen);

        for (size_t i = 0; i < 1000; ++i) {
            TString str(" ");
            str += ToString(i % 10);

            bo.Write(str.data(), str.size());
        }

        bo.Finish();
    }

    UNIT_ASSERT_EQUAL(s.size(), 2000);

    {
        const size_t buflen = 11;
        TBuffered<TStringInput> bi(buflen, s);

        for (size_t i = 0; i < 1000; ++i) {
            TString str(" ");
            str += ToString(i % 10);

            char buf[3];

            UNIT_ASSERT_EQUAL(bi.Load(buf, 2), 2);

            buf[2] = 0;

            UNIT_ASSERT_EQUAL(str, buf);
        }
    }

    s.clear();

    {
        const size_t buflen = 13;
        TBuffered<TMyStringOutput> bo(buflen, s, buflen);
        TString f = "1234567890";

        for (size_t i = 0; i < 10; ++i) {
            f += f;
        }

        for (size_t i = 0; i < 1000; ++i) {
            bo.Write(f.data(), i);
        }

        bo.Finish();
    }
}

void TStreamsTest::TestBufferStream() {
    TBufferStream stream;
    TString s = "test";

    stream.Write(s.data(), s.size());
    char buf[5];
    size_t bytesRead = stream.Read(buf, 4);
    UNIT_ASSERT_EQUAL(4, bytesRead);
    UNIT_ASSERT_EQUAL(0, strncmp(s.data(), buf, 4));

    stream.Write(s.data(), s.size());
    bytesRead = stream.Read(buf, 2);
    UNIT_ASSERT_EQUAL(2, bytesRead);
    UNIT_ASSERT_EQUAL(0, strncmp("te", buf, 2));

    bytesRead = stream.Read(buf, 2);
    UNIT_ASSERT_EQUAL(2, bytesRead);
    UNIT_ASSERT_EQUAL(0, strncmp("st", buf, 2));

    bytesRead = stream.Read(buf, 2);
    UNIT_ASSERT_EQUAL(0, bytesRead);
}

namespace {
    class TStringListInput: public IWalkInput {
    public:
        TStringListInput(const TVector<TString>& data)
            : Data_(data)
            , Index_(0)
        {
        }

    protected:
        size_t DoUnboundedNext(const void** ptr) override {
            if (Index_ >= Data_.size()) {
                return 0;
            }

            const TString& string = Data_[Index_++];

            *ptr = string.data();
            return string.size();
        }

    private:
        const TVector<TString>& Data_;
        size_t Index_;
    };

    const char Text[] =
        // UTF8 encoded "one \ntwo\r\nthree\n\tfour\nfive\n" in russian and ...
        "один \n"
        "два\r\n"
        "три\n"
        "\tчетыре\n"
        "пять\n"
        // ... additional test cases
        "\r\n"
        "\n\r" // this char goes to the front of the next string
        "one two\n"
        "123\r\n"
        "\t\r ";

    const char* Expected[] = {
        // UTF8 encoded "one ", "two", "three", "\tfour", "five" in russian and ...
        "один ",
        "два",
        "три",
        "\tчетыре",
        "пять",
        // ... additional test cases
        "",
        "",
        "\rone two",
        "123",
        "\t\r "};
    void TestStreamReadTo1(IInputStream& input, const char* comment) {
        TString tmp;
        input.ReadTo(tmp, 'c');
        UNIT_ASSERT_VALUES_EQUAL_C(tmp, "111a222b333", comment);

        char tmp2;
        input.Read(&tmp2, 1);
        UNIT_ASSERT_VALUES_EQUAL_C(tmp2, '4', comment);

        input.ReadTo(tmp, '6');
        UNIT_ASSERT_VALUES_EQUAL_C(tmp, "44d555e", comment);

        tmp = input.ReadAll();
        UNIT_ASSERT_VALUES_EQUAL_C(tmp, "66f", comment);
    }

    void TestStreamReadTo2(IInputStream& input, const char* comment) {
        TString s;
        size_t i = 0;
        while (input.ReadLine(s)) {
            UNIT_ASSERT_C(i < Y_ARRAY_SIZE(Expected), comment);
            UNIT_ASSERT_VALUES_EQUAL_C(s, Expected[i], comment);
            ++i;
        }
    }

    void TestStreamReadTo3(IInputStream& input, const char* comment) {
        UNIT_ASSERT_VALUES_EQUAL_C(input.ReadLine(), "111a222b333c444d555e666f", comment);
    }

    void TestStreamReadTo4(IInputStream& input, const char* comment) {
        UNIT_ASSERT_VALUES_EQUAL_C(input.ReadTo('\0'), "one", comment);
        UNIT_ASSERT_VALUES_EQUAL_C(input.ReadTo('\0'), "two", comment);
        UNIT_ASSERT_VALUES_EQUAL_C(input.ReadTo('\0'), "three", comment);
    }

    void TestStrokaInput(IInputStream& input, const char* comment) {
        TString line;
        ui32 i = 0;
        TInstant start = Now();
        while (input.ReadLine(line)) {
            ++i;
        }
        Cout << comment << ":" << (Now() - start).SecondsFloat() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(i, 100000);
    }

    template <class T>
    void TestStreamReadTo(const TString& text, T test) {
        TStringInput is(text);
        test(is, "TStringInput");
        TMemoryInput mi(text.data(), text.size());
        test(mi, "TMemoryInput");
        TBuffer b(text.data(), text.size());
        TBufferInput bi(b);
        test(bi, "TBufferInput");
        TStringInput slave(text);
        TBufferedInput bdi(&slave);
        test(bdi, "TBufferedInput");
        TVector<TString> lst(1, text);
        TStringListInput sli(lst);
        test(sli, "IWalkInput");
    }
} // namespace

void TStreamsTest::TestReadTo() {
    TestStreamReadTo("111a222b333c444d555e666f", TestStreamReadTo1);
    TestStreamReadTo(Text, TestStreamReadTo2);
    TestStreamReadTo("111a222b333c444d555e666f", TestStreamReadTo3);
    TString withZero = "one";
    withZero.append('\0').append("two").append('\0').append("three");
    TestStreamReadTo(withZero, TestStreamReadTo4);
}

void TStreamsTest::TestStrokaInput() {
    TString s;
    for (ui32 i = 0; i < 100000; ++i) {
        TVector<char> d(i % 1000, 'a');
        s.append(d.data(), d.size());
        s.append('\n');
    }
    TestStreamReadTo(s, ::TestStrokaInput);
}

void TStreamsTest::TestWtrokaInput() {
    const TString s(Text);
    TStringInput is(s);
    TUtf16String w;
    size_t i = 0;

    while (is.ReadLine(w)) {
        UNIT_ASSERT(i < Y_ARRAY_SIZE(Expected));
        UNIT_ASSERT_VALUES_EQUAL(w, UTF8ToWide(Expected[i]));

        ++i;
    }
}

void TStreamsTest::TestWtrokaOutput() {
    TString s;
    TStringOutput os(s);
    const size_t n = sizeof(Expected) / sizeof(Expected[0]);

    for (size_t i = 0; i < n; ++i) {
        TUtf16String w = UTF8ToWide(Expected[i]);

        os << w;

        if (i == 1 || i == 5 || i == 8) {
            os << '\r';
        }

        if (i < n - 1) {
            os << '\n';
        }
    }

    UNIT_ASSERT(s == Text);
}

void TStreamsTest::TestWchar16Output() {
    TString s;
    TStringOutput os(s);
    os << wchar16(97); // latin a
    os << u'\u044E';   // cyrillic ю
    os << u'я';
    os << wchar16(0xD801); // high surrogate is printed as replacement character U+FFFD
    os << u'b';

    UNIT_ASSERT_VALUES_EQUAL(s, "aюя"
                                "\xEF\xBF\xBD"
                                "b");
}

void TStreamsTest::TestWchar32Output() {
    TString s;
    TStringOutput os(s);
    os << wchar32(97); // latin a
    os << U'\u044E';   // cyrillic ю
    os << U'я';
    os << U'\U0001F600'; // grinning face
    os << u'b';

    UNIT_ASSERT_VALUES_EQUAL(s, "aюя"
                                "\xF0\x9F\x98\x80"
                                "b");
}

void TStreamsTest::TestUtf16StingOutputByChars() {
    TString s = "\xd1\x87\xd0\xb8\xd1\x81\xd1\x82\xd0\xb8\xd1\x87\xd0\xb8\xd1\x81\xd1\x82\xd0\xb8";
    TUtf16String w = UTF8ToWide(s);

    UNIT_ASSERT_VALUES_EQUAL(w.size(), 10);

    TStringStream stream0;
    stream0 << w;
    UNIT_ASSERT_VALUES_EQUAL(stream0.Str(), s);

    TStringStream stream1;
    for (size_t i = 0; i < 10; i++) {
        stream1 << w[i];
    }
    UNIT_ASSERT_VALUES_EQUAL(stream1.Str(), s);
}
