#include <util/generic/set.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <utility>

#include <library/cpp/http/io/headers.h>
#include <library/cpp/testing/unittest/registar.h>

namespace {
    class THeadersExistence {
    public:
        THeadersExistence() = default;

        THeadersExistence(const THttpHeaders& headers) {
            for (THttpHeaders::TConstIterator it = headers.Begin();
                 it != headers.End();
                 ++it) {
                Add(it->Name(), it->Value());
            }
        }

    public:
        void Add(TStringBuf name, TStringBuf value) {
            Impl.emplace(TString(name), TString(value));
        }

        bool operator==(const THeadersExistence& rhs) const {
            return Impl == rhs.Impl;
        }

    private:
        typedef TMultiSet<std::pair<TString, TString>> TImpl;
        TImpl Impl;
    };
}

bool operator==(const THeadersExistence& lhs, const THttpHeaders& rhs) {
    return lhs == THeadersExistence(rhs);
}

bool operator==(const THttpHeaders& lhs, const THeadersExistence& rhs) {
    return THeadersExistence(lhs) == rhs;
}

class THttpHeadersTest: public TTestBase {
    UNIT_TEST_SUITE(THttpHeadersTest);
    UNIT_TEST(TestConstructorFromArrayRef);
    UNIT_TEST(TestAddOperation1Arg);
    UNIT_TEST(TestAddOperation2Args);
    UNIT_TEST(TestAddOrReplaceOperation1Arg);
    UNIT_TEST(TestAddOrReplaceOperation2Args);
    UNIT_TEST(TestAddHeaderTemplateness);
    UNIT_TEST(TestFindHeader);
    UNIT_TEST_SUITE_END();

private:
    typedef void (*TAddHeaderFunction)(THttpHeaders&, TStringBuf name, TStringBuf value);
    typedef void (*TAddOrReplaceHeaderFunction)(THttpHeaders&, TStringBuf name, TStringBuf value);

public:
    void TestConstructorFromArrayRef();
    void TestAddOperation1Arg();
    void TestAddOperation2Args();
    void TestAddOrReplaceOperation1Arg();
    void TestAddOrReplaceOperation2Args();
    void TestAddHeaderTemplateness();
    void TestFindHeader();

private:
    static void AddHeaderImpl1Arg(THttpHeaders& headers, TStringBuf name, TStringBuf value) {
        headers.AddHeader(THttpInputHeader(TString(name), TString(value)));
    }

    static void AddHeaderImpl2Args(THttpHeaders& headers, TStringBuf name, TStringBuf value) {
        headers.AddHeader(TString(name), TString(value));
    }

    static void AddOrReplaceHeaderImpl1Arg(THttpHeaders& headers, TStringBuf name, TStringBuf value) {
        headers.AddOrReplaceHeader(THttpInputHeader(TString(name), TString(value)));
    }

    static void AddOrReplaceHeaderImpl2Args(THttpHeaders& headers, TStringBuf name, TStringBuf value) {
        headers.AddOrReplaceHeader(TString(name), TString(value));
    }

    void DoTestAddOperation(TAddHeaderFunction);
    void DoTestAddOrReplaceOperation(TAddHeaderFunction, TAddOrReplaceHeaderFunction);
};

UNIT_TEST_SUITE_REGISTRATION(THttpHeadersTest);

void THttpHeadersTest::TestConstructorFromArrayRef() {
    THeadersExistence expected;
    expected.Add("h1", "v1");
    expected.Add("h2", "v2");

    // Construct from vector
    TVector<THttpInputHeader> headerVec{
        {"h1", "v1"},
        {"h2", "v2"}
    };
    THttpHeaders h1(headerVec);
    UNIT_ASSERT(expected == h1);

    // Construct from initializer list
    THttpHeaders h2({
        {"h1", "v1"},
        {"h2", "v2"}
    });
    UNIT_ASSERT(expected == h2);
}
void THttpHeadersTest::TestAddOperation1Arg() {
    DoTestAddOperation(AddHeaderImpl1Arg);
}
void THttpHeadersTest::TestAddOperation2Args() {
    DoTestAddOperation(AddHeaderImpl2Args);
}

void THttpHeadersTest::TestAddOrReplaceOperation1Arg() {
    DoTestAddOrReplaceOperation(AddHeaderImpl1Arg, AddOrReplaceHeaderImpl1Arg);
}
void THttpHeadersTest::TestAddOrReplaceOperation2Args() {
    DoTestAddOrReplaceOperation(AddHeaderImpl2Args, AddOrReplaceHeaderImpl2Args);
}

void THttpHeadersTest::DoTestAddOperation(TAddHeaderFunction addHeader) {
    THttpHeaders h1;

    addHeader(h1, "h1", "v1");
    addHeader(h1, "h2", "v1");

    addHeader(h1, "h3", "v1");
    addHeader(h1, "h3", "v2");
    addHeader(h1, "h3", "v2");

    THeadersExistence h2;

    h2.Add("h1", "v1");
    h2.Add("h2", "v1");

    h2.Add("h3", "v1");
    h2.Add("h3", "v2");
    h2.Add("h3", "v2");

    UNIT_ASSERT(h2 == h1);
}

// Sorry, but AddOrReplaceHeader replaces only first occurence
void THttpHeadersTest::DoTestAddOrReplaceOperation(TAddHeaderFunction addHeader, TAddOrReplaceHeaderFunction addOrReplaceHeader) {
    THttpHeaders h1;

    addHeader(h1, "h1", "v1");

    addOrReplaceHeader(h1, "h2", "v1");
    addOrReplaceHeader(h1, "h2", "v2");
    addOrReplaceHeader(h1, "h2", "v3");
    addHeader(h1, "h2", "v4");

    addHeader(h1, "h3", "v1");
    addHeader(h1, "h3", "v2");
    addOrReplaceHeader(h1, "h3", "v3");

    THeadersExistence h2;

    h2.Add("h1", "v1");

    h2.Add("h2", "v3");
    h2.Add("h2", "v4");

    h2.Add("h3", "v2");
    h2.Add("h3", "v3");

    UNIT_ASSERT(h2 == h1);
}

void THttpHeadersTest::TestAddHeaderTemplateness() {
    THttpHeaders h1;
    h1.AddHeader("h1", "v1");
    h1.AddHeader("h2", TString("v2"));
    h1.AddHeader("h3", TStringBuf("v3"));
    h1.AddHeader("h4", TStringBuf("v4"));

    THeadersExistence h2;
    h2.Add("h1", "v1");
    h2.Add("h2", "v2");
    h2.Add("h3", "v3");
    h2.Add("h4", "v4");

    UNIT_ASSERT(h1 == h2);
}

void THttpHeadersTest::TestFindHeader() {
    THttpHeaders sut;
    sut.AddHeader("NaMe", "Value");

    UNIT_ASSERT(sut.FindHeader("name"));
    UNIT_ASSERT(sut.FindHeader("name")->Value() == "Value");
}
