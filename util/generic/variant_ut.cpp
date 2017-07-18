#include "variant.h"

#include <library/unittest/registar.h>
#include <util/generic/string.h>

namespace {
    struct S {
        static int CtorCalls;
        static int DtorCalls;
        static int CopyCtorCalls;
        static int MoveCtorCalls;

        static void Reset() {
            CtorCalls = 0;
            DtorCalls = 0;
            CopyCtorCalls = 0;
            MoveCtorCalls = 0;
        }

        int Value;

        explicit S(int value)
            : Value(value)
        {
            ++CtorCalls;
        }

        S(const S& other)
            : Value(other.Value)
        {
            ++CopyCtorCalls;
        }

        S(S&& other)
            : Value(other.Value)
        {
            other.Value = -1;
            ++MoveCtorCalls;
        }

        ~S() {
            ++DtorCalls;
        }
    };

    int S::CtorCalls;
    int S::DtorCalls;
    int S::CopyCtorCalls;
    int S::MoveCtorCalls;

    class TNonCopyable1
       : private TNonCopyable {
    public:
        TNonCopyable1() {
        }
    };

    class TNonCopyable2
       : private TNonCopyable {
    public:
        TNonCopyable2() {
        }
    };

    struct TVisitorToString {
        TString operator()(const TString& s) const {
            return s;
        }
        TString operator()(const int& x) const {
            return ToString(x);
        }
    };

    struct TVisitorIncrement {
        void operator()(TString& s) const {
            s += "1";
        }
        void operator()(int& x) const {
            x += 1;
        }
    };

} // namespace

class TVariantTest: public TTestBase {
    UNIT_TEST_SUITE(TVariantTest);
    UNIT_TEST(TestPod1);
    UNIT_TEST(TestPod2);
    UNIT_TEST(TestNonPod1);
    UNIT_TEST(TestNonPod2);
    UNIT_TEST(TestConstructCopy1);
    UNIT_TEST(TestConstructCopy2);
    UNIT_TEST(TestConstructMove1);
    UNIT_TEST(TestConstructMove2);
    UNIT_TEST(TestMove);
    UNIT_TEST(TestAssignCopy);
    UNIT_TEST(TestMoveCopy);
    UNIT_TEST(TestInplace);
    UNIT_TEST(TestEquals);
    UNIT_TEST(TestVisitor);
    UNIT_TEST(TestHash);
    UNIT_TEST_SUITE_END();

private:
    void TestPod1() {
        TVariant<int> v(123);

        UNIT_ASSERT(v.Is<int>());

        UNIT_ASSERT_EQUAL(0, v.Tag());
        UNIT_ASSERT_EQUAL(0, v.TagOf<int>());

        UNIT_ASSERT_EQUAL(123, v.As<int>());
        UNIT_ASSERT_EQUAL(123, *v.TryAs<int>());
    }

    void TestPod2() {
        TVariant<int, double> v(3.14);

        UNIT_ASSERT(v.Is<double>());
        UNIT_ASSERT(!v.Is<int>());

        UNIT_ASSERT_EQUAL(1, v.Tag());
        UNIT_ASSERT_EQUAL(0, v.TagOf<int>());
        UNIT_ASSERT_EQUAL(1, v.TagOf<double>());

        UNIT_ASSERT_EQUAL(3.14, v.As<double>());
        UNIT_ASSERT_EQUAL(3.14, *v.TryAs<double>());
        UNIT_ASSERT_EQUAL(nullptr, v.TryAs<int>());
    }

    void TestNonPod1() {
        TVariant<TString> v(TString("hello"));
        UNIT_ASSERT_EQUAL("hello", v.As<TString>());
    }

    void TestNonPod2() {
        S::Reset();
        {
            TVariant<TString, S> v(TString("hello"));
            UNIT_ASSERT_EQUAL("hello", v.As<TString>());
        }
        UNIT_ASSERT_EQUAL(0, S::CtorCalls);
        UNIT_ASSERT_EQUAL(0, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestConstructCopy1() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v(s);
            UNIT_ASSERT_EQUAL(123, v.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(2, S::DtorCalls);
        UNIT_ASSERT_EQUAL(1, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestConstructCopy2() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            TVariant<TString, S> v2(v1);
            UNIT_ASSERT_EQUAL(123, s.Value);
            UNIT_ASSERT_EQUAL(123, v1.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(2, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestConstructMove1() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v(std::move(s));
            UNIT_ASSERT_EQUAL(123, v.As<S>().Value);
            UNIT_ASSERT_EQUAL(-1, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(2, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestConstructMove2() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(std::move(s));
            TVariant<TString, S> v2(std::move(v1));
            UNIT_ASSERT_EQUAL(-1, v1.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.As<S>().Value);
            UNIT_ASSERT_EQUAL(-1, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(2, S::MoveCtorCalls);
    }

    void TestMove() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            UNIT_ASSERT_EQUAL(123, v1.As<S>().Value);

            TVariant<TString, S> v2(std::move(v1));
            UNIT_ASSERT_EQUAL(-1, v1.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(1, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestAssignCopy() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            TVariant<TString, S> v2(TString("hello"));
            v2 = v1;

            UNIT_ASSERT_EQUAL(123, v1.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(2, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestMoveCopy() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            TVariant<TString, S> v2(TString("hello"));
            v2 = std::move(v1);

            UNIT_ASSERT_EQUAL(-1, v1.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.As<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(1, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestInplace() {
        TVariant<TNonCopyable1, TNonCopyable2> v1{TVariantTypeTag<TNonCopyable1>()};
        UNIT_ASSERT(v1.Is<TNonCopyable1>());
        UNIT_ASSERT(!v1.Is<TNonCopyable2>());

        TVariant<TNonCopyable1, TNonCopyable2> v2{TVariantTypeTag<TNonCopyable2>()};
        UNIT_ASSERT(!v2.Is<TNonCopyable1>());
        UNIT_ASSERT(v2.Is<TNonCopyable2>());
    }

    void TestEquals() {
        const TVariant<int, TString> varInt1(10);
        const TVariant<int, TString> varInt2(10);
        const TVariant<int, TString> varInt3(20);
        UNIT_ASSERT_EQUAL(varInt1, 10);
        UNIT_ASSERT_UNEQUAL(varInt1, 20);
        UNIT_ASSERT_EQUAL(varInt1, varInt1);
        UNIT_ASSERT_EQUAL(varInt1, varInt2);
        UNIT_ASSERT_UNEQUAL(varInt1, varInt3);

        const TVariant<int, TString> varStr1(TString("hello"));
        const TVariant<int, TString> varStr2(TString("hello"));
        const TVariant<int, TString> varStr3(TString("world"));
        UNIT_ASSERT_EQUAL(varStr1, TString("hello"));
        UNIT_ASSERT_UNEQUAL(varStr1, TString("world"));
        UNIT_ASSERT_EQUAL(varStr1, varStr1);
        UNIT_ASSERT_EQUAL(varStr1, varStr2);
        UNIT_ASSERT_UNEQUAL(varStr1, varStr3);

        UNIT_ASSERT_UNEQUAL(varInt1, varStr1);
        UNIT_ASSERT_UNEQUAL(varInt1, TString("hello"));
        UNIT_ASSERT_UNEQUAL(varStr1, varInt1);
        UNIT_ASSERT_UNEQUAL(varStr1, 10);
    }

    void TestVisitor() {
        TVariant<int, TString> varInt(10);
        UNIT_ASSERT_EQUAL(varInt.Visit(TVisitorToString()), "10");
        TVariant<int, TString> varStr(TString("hello"));
        UNIT_ASSERT_EQUAL(varStr.Visit(TVisitorToString()), "hello");

        varInt.Visit(TVisitorIncrement());
        UNIT_ASSERT_EQUAL(varInt, 11);
        varStr.Visit(TVisitorIncrement());
        UNIT_ASSERT_EQUAL(varStr, TString("hello1"));
    }

    void TestHash() {
        using TIntOrStr = TVariant<int, TString>;
        yhash_set<TIntOrStr> set;
        set.insert(TIntOrStr(1));
        set.insert(TIntOrStr(2));
        set.insert(TIntOrStr(100));
        set.insert(TIntOrStr(TString("hello")));
        set.insert(TIntOrStr(TString("world")));
        set.insert(TIntOrStr(TString("qwerty")));

        UNIT_ASSERT(set.has(TIntOrStr(1)));
        UNIT_ASSERT(set.has(TIntOrStr(TString("hello"))));
        UNIT_ASSERT(!set.has(TIntOrStr(321)));
        UNIT_ASSERT(!set.has(TIntOrStr(TString("asdfgh"))));

        UNIT_ASSERT_EQUAL(set.size(), 6);
        set.insert(TIntOrStr(1));
        set.insert(TIntOrStr(TString("hello")));
        UNIT_ASSERT_EQUAL(set.size(), 6);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TVariantTest)
