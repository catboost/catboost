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

        S& operator=(const S& rhs) {
            Value = rhs.Value;
            ++CopyCtorCalls;
            return *this;
        }

        S& operator=(S&& rhs) {
            Value = rhs.Value;
            rhs.Value = -1;
            ++MoveCtorCalls;
            return *this;
        }

        bool operator== (const S& s) const {
            return Value == s.Value;
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

    struct TVisitorDouble {
        TVariant<int, TString> operator()(int x) const {
            return x * 2;
        }

        TVariant<int, TString> operator()(const TString& s) const {
            return s + s;
        }
    };

}

template <>
struct THash<S> {
    size_t operator()(const S& s) const {
        UNIT_ASSERT(s.Value != -1);
        return s.Value;
    }
};


class TVariantTest: public TTestBase {
    UNIT_TEST_SUITE(TVariantTest);
    UNIT_TEST(TestStatic);
    UNIT_TEST(TestPod1);
    UNIT_TEST(TestPod2);
    UNIT_TEST(TestNonPod1);
    UNIT_TEST(TestNonPod2);
    UNIT_TEST(TestTypeTag);
    UNIT_TEST(TestTypeTagWithoutArgs);
    UNIT_TEST(TestIndexTag);
    UNIT_TEST(TestIndexTagWithoutArgs);
    UNIT_TEST(TestConstructCopy1);
    UNIT_TEST(TestConstructCopy2);
    UNIT_TEST(TestConstructMove1);
    UNIT_TEST(TestConstructMove2);
    UNIT_TEST(TestMove);
    UNIT_TEST(TestCopyAssign);
    UNIT_TEST(TestMoveAssign);
    UNIT_TEST(TestInplace);
    UNIT_TEST(TestEmplace);
    UNIT_TEST(TestDefaultConstructor);
    UNIT_TEST(TestMonostate);
    UNIT_TEST(TestEquals);
    UNIT_TEST(TestVisitor);
    UNIT_TEST(TestHash);
    UNIT_TEST(TestVisitorWithoutDefaultConstructor);
    UNIT_TEST(TestSwapSameAlternative);
    UNIT_TEST(TestSwapDifferentAlternative);
    UNIT_TEST(TestGetThrow);
    UNIT_TEST(TestLvalueVisit);
    UNIT_TEST(TestRvalueVisit);
    UNIT_TEST_SUITE_END();

private:
    void TestStatic() {
        using TV1 = TVariant<int, double, short>;
        using TV2 = TVariant<TString, double>;
        using TV3 = TVariant<S, TNonCopyable>;
        using TV4 = TVariant<TNonCopyable1, int>;
        using TV5 = TVariant<TMonostate>;
        using TV6 = TVariant<TMonostate, size_t>;

        static_assert(std::is_same<TVariantAlternative<0, TV1>::type, int>::value, "");
        static_assert(std::is_same<TVariantAlternative<1, TV1>::type, double>::value, "");
        static_assert(std::is_same<TVariantAlternative<2, TV1>::type, short>::value, "");
        static_assert(TVariantSize<TV1>::value == 3, "");

        static_assert(std::is_same<TVariantAlternative<0, TV2>::type, TString>::value, "");
        static_assert(std::is_same<TVariantAlternative<1, TV2>::type, double>::value, "");
        static_assert(TVariantSize<TV2>::value == 2, "");

        static_assert(std::is_same<TVariantAlternative<0, TV3>::type, S>::value, "");
        static_assert(std::is_same<TVariantAlternative<1, TV3>::type, TNonCopyable>::value, "");
        static_assert(TVariantSize<TV3>::value == 2, "");

        static_assert(std::is_same<TVariantAlternative<0, TV4>::type, TNonCopyable1>::value, "");
        static_assert(std::is_same<TVariantAlternative<1, TV4>::type, int>::value, "");
        static_assert(TVariantSize<TV4>::value == 2, "");

        static_assert(std::is_same<TVariantAlternative<0, TV5>::type, TMonostate>::value, "");
        static_assert(TVariantSize<TV5>::value == 1, "");

        static_assert(std::is_same<TVariantAlternative<0, TV6>::type, TMonostate>::value, "");
        static_assert(std::is_same<TVariantAlternative<1, TV6>::type, size_t>::value, "");
        static_assert(TVariantSize<TV6>::value == 2, "");
    }

    void TestPod1() {
        TVariant<int> v(123);

        UNIT_ASSERT(v.HoldsAlternative<int>());

        UNIT_ASSERT_EQUAL(0, v.Index());
        UNIT_ASSERT_EQUAL(0, v.TagOf<int>());

        UNIT_ASSERT_EQUAL(123, v.Get<int>());
        UNIT_ASSERT(v.GetIf<int>());
        UNIT_ASSERT_EQUAL(123, *v.GetIf<int>());
    }

    void TestPod2() {
        TVariant<int, double> v(3.14);

        UNIT_ASSERT(v.HoldsAlternative<double>());
        UNIT_ASSERT(!v.HoldsAlternative<int>());

        UNIT_ASSERT_EQUAL(1, v.Index());
        UNIT_ASSERT_EQUAL(0, v.TagOf<int>());
        UNIT_ASSERT_EQUAL(1, v.TagOf<double>());

        UNIT_ASSERT_EQUAL(3.14, v.Get<double>());
        UNIT_ASSERT(v.GetIf<double>());
        UNIT_ASSERT_EQUAL(3.14, *v.GetIf<double>());
        UNIT_ASSERT(!v.GetIf<int>());
    }

    void TestNonPod1() {
        TVariant<TString> v(TString("hello"));
        UNIT_ASSERT_EQUAL("hello", v.Get<TString>());
    }

    void TestNonPod2() {
        S::Reset();
        {
            TVariant<TString, S> v(TString("hello"));
            UNIT_ASSERT_EQUAL("hello", v.Get<TString>());
        }
        UNIT_ASSERT_EQUAL(0, S::CtorCalls);
        UNIT_ASSERT_EQUAL(0, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestTypeTag() {
        S::Reset();
        {
            TVariant<TString, S, int> v(TVariantTypeTag<S>{}, 6);
            UNIT_ASSERT(v.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(1, v.Index());
            UNIT_ASSERT_EQUAL(6, v.Get<S>().Value);
            UNIT_ASSERT_EQUAL(6, v.Get<1>().Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(1, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestTypeTagWithoutArgs() {
        S::Reset();
        {
            TVariant<S, TString, int> v(TVariantTypeTag<TString>{});
            UNIT_ASSERT(v.HoldsAlternative<TString>());
            UNIT_ASSERT(v.Get<TString>().Empty());
            UNIT_ASSERT(v.Get<1>().Empty());
            UNIT_ASSERT_EQUAL(1, v.Index());
        }
        UNIT_ASSERT_EQUAL(0, S::CtorCalls);
        UNIT_ASSERT_EQUAL(0, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestIndexTag() {
        S::Reset();
        {
            TVariant<TString, double, S, int> v(TVariantIndexTag<2>{}, 25);
            UNIT_ASSERT(v.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(2, v.Index());
            UNIT_ASSERT_EQUAL(25, v.Get<S>().Value);
            UNIT_ASSERT_EQUAL(25, v.Get<2>().Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(1, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestIndexTagWithoutArgs() {
        S::Reset();
        {
            TVariant<TString, double, S, int> v(TVariantIndexTag<0>{});
            UNIT_ASSERT(v.HoldsAlternative<TString>());
            UNIT_ASSERT(v.Get<TString>().Empty());
            UNIT_ASSERT(v.Get<0>().Empty());
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
            UNIT_ASSERT_EQUAL(123, v.Get<S>().Value);
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
            UNIT_ASSERT_EQUAL(123, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.Get<S>().Value);
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
            UNIT_ASSERT_EQUAL(123, v.Get<S>().Value);
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
            UNIT_ASSERT_EQUAL(-1, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.Get<S>().Value);
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
            UNIT_ASSERT_EQUAL(123, v1.Get<S>().Value);

            TVariant<TString, S> v2(std::move(v1));
            UNIT_ASSERT_EQUAL(-1, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(1, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestCopyAssign() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            TVariant<TString, S> v2(TString("hello"));
            v2 = v1;

            UNIT_ASSERT_EQUAL(123, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(2, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(0, S::MoveCtorCalls);
    }

    void TestMoveAssign() {
        S::Reset();
        {
            S s(123);
            TVariant<TString, S> v1(s);
            UNIT_ASSERT(v1.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(1, v1.Index());
            TVariant<TString, S> v2(TString("hello"));
            UNIT_ASSERT(v2.HoldsAlternative<TString>());
            UNIT_ASSERT_EQUAL(0, v2.Index());

            v2 = std::move(v1);

            UNIT_ASSERT_EQUAL(-1, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, v2.Get<S>().Value);
            UNIT_ASSERT_EQUAL(123, s.Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(1, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestInplace() {
        using TVar = TVariant<TNonCopyable1, TNonCopyable2>;
        TVar v1{TVariantTypeTag<TNonCopyable1>()};
        UNIT_ASSERT(v1.HoldsAlternative<TNonCopyable1>());
        UNIT_ASSERT(!v1.HoldsAlternative<TNonCopyable2>());

        TVar v2{TVariantTypeTag<TNonCopyable2>()};
        UNIT_ASSERT(!v2.HoldsAlternative<TNonCopyable1>());
        UNIT_ASSERT(v2.HoldsAlternative<TNonCopyable2>());

        TVar v3{TVariantIndexTag<0>()};
        UNIT_ASSERT(v1.HoldsAlternative<TNonCopyable1>());
        UNIT_ASSERT(!v1.HoldsAlternative<TNonCopyable2>());

        TVar v4{TVariantIndexTag<1>()};
        UNIT_ASSERT(!v2.HoldsAlternative<TNonCopyable1>());
        UNIT_ASSERT(v2.HoldsAlternative<TNonCopyable2>());
    }

    void TestEmplace() {
        S::Reset();
        {
            TVariant<TString, S> var{TVariantTypeTag<S>(), 222};
            UNIT_ASSERT(var.HoldsAlternative<S>());
            UNIT_ASSERT_VALUES_EQUAL(222, var.Get<S>().Value);
            UNIT_ASSERT_EQUAL(1, S::CtorCalls);
            var.Emplace<TString>("foobar");
            UNIT_ASSERT_VALUES_EQUAL("foobar", var.Get<TString>());
            UNIT_ASSERT_EQUAL(1, S::DtorCalls);
            var.Emplace<S>(333);
            UNIT_ASSERT_EQUAL(2, S::CtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(333, var.Get<S>().Value);
            var.Emplace<S>(444);
            UNIT_ASSERT_EQUAL(3, S::CtorCalls);
            UNIT_ASSERT_EQUAL(2, S::DtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(444, var.Get<S>().Value);
        }
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
    }

    void TestDefaultConstructor() {
        S::Reset();
        {
            TVariant<TMonostate, S> var;
            UNIT_ASSERT(0 == var.Index());
            UNIT_ASSERT_VALUES_EQUAL(0, S::CtorCalls);
            var.Emplace<S>(123);
            UNIT_ASSERT_VALUES_EQUAL(1, S::CtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(0, S::DtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(123, var.Get<S>().Value);
            UNIT_ASSERT_VALUES_EQUAL(var.TagOf<S>(), var.Index());
            var.Emplace<0>();
            UNIT_ASSERT_VALUES_EQUAL(1, S::CtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(1, S::DtorCalls);
            var = S(321);
            UNIT_ASSERT_VALUES_EQUAL(2, S::CtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(1, S::MoveCtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(2, S::DtorCalls);
            UNIT_ASSERT_VALUES_EQUAL(321, var.Get<S>().Value);
            UNIT_ASSERT_VALUES_EQUAL(var.TagOf<S>(), var.Index());
        }
        UNIT_ASSERT_VALUES_EQUAL(1, S::MoveCtorCalls);
        UNIT_ASSERT_VALUES_EQUAL(2, S::CtorCalls);
        UNIT_ASSERT_VALUES_EQUAL(3, S::DtorCalls);
    }

    void TestMonostate() {
        using TVar = TVariant<TMonostate>;
        TVar var;
        UNIT_ASSERT(0 == var.Index());
        UNIT_ASSERT_EQUAL(1, TVariantSize<TVar>::value);
        UNIT_ASSERT_EQUAL(1, TVariantSize<decltype(var)>::value);
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
        THashSet<TIntOrStr> set;
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

    void TestVisitorWithoutDefaultConstructor() {
        TVariant<int, TString> varInt(10);
        UNIT_ASSERT_EQUAL(varInt.Visit(TVisitorDouble()), 20);
        TVariant<int, TString> varStr(TString("hello"));
        UNIT_ASSERT_EQUAL(varStr.Visit(TVisitorDouble()), TString("hellohello"));
    }

    void TestSwapSameAlternative() {
        S::Reset();
        {
            TVariant<TString, S> v1(TVariantTypeTag<S>{}, 5);
            TVariant<TString, S> v2(TVariantTypeTag<S>{}, 64);
            UNIT_ASSERT(v1.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(5, v1.Get<S>().Value);
            UNIT_ASSERT(v2.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(64, v2.Get<S>().Value);
            v1.Swap(v2);
            UNIT_ASSERT_EQUAL(64, v1.Get<S>().Value);
            UNIT_ASSERT_EQUAL(5, v2.Get<S>().Value);
        }
        UNIT_ASSERT_EQUAL(2, S::CtorCalls);
        UNIT_ASSERT_EQUAL(3, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(3, S::MoveCtorCalls);
    }

    void TestSwapDifferentAlternative() {
        S::Reset();
        {
            TVariant<TString, S> v1(TVariantTypeTag<S>{}, 5);
            TVariant<TString, S> v2(TVariantTypeTag<TString>{}, "test");
            UNIT_ASSERT(v1.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(5, v1.Get<S>().Value);
            UNIT_ASSERT(v2.HoldsAlternative<TString>());
            UNIT_ASSERT_EQUAL("test", v2.Get<TString>());
            v1.Swap(v2);
            UNIT_ASSERT(v1.HoldsAlternative<TString>());
            UNIT_ASSERT_EQUAL("test", v1.Get<TString>());
            UNIT_ASSERT(v2.HoldsAlternative<S>());
            UNIT_ASSERT_EQUAL(5, v2.Get<S>().Value);
        }
        UNIT_ASSERT_EQUAL(1, S::CtorCalls);
        UNIT_ASSERT_EQUAL(2, S::DtorCalls);
        UNIT_ASSERT_EQUAL(0, S::CopyCtorCalls);
        UNIT_ASSERT_EQUAL(1, S::MoveCtorCalls);
    }

    void TestGetThrow() {
        TVariant<int, double, TString> v(TVariantIndexTag<0>(), 1);
        UNIT_ASSERT(v.HoldsAlternative<int>());
        UNIT_ASSERT_EQUAL(0, v.Index());
        UNIT_ASSERT_EXCEPTION(v.Get<1>(), TWrongVariantError);
        UNIT_ASSERT_EXCEPTION(v.Get<2>(), TWrongVariantError);
        UNIT_ASSERT_EXCEPTION(v.Get<double>(), TWrongVariantError);
        UNIT_ASSERT_EXCEPTION(v.Get<TString>(), TWrongVariantError);
    }

    void TestLvalueVisit() {
        TVariant<int, TString> v;
        v.Get<int>() = 5;
        TVisitorToString vis;
        UNIT_ASSERT_EQUAL("5", Visit(vis, v));
    }

    void TestRvalueVisit() {
        TVariant<int, TString> v;
        v.Get<int>() = 6;
        UNIT_ASSERT_EQUAL("6", Visit(TVisitorToString{}, v));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TVariantTest)
