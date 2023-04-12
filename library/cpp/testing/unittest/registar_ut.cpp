#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TUnitTestMacroTest) {
    Y_UNIT_TEST(Assert) {
        auto unitAssert = [] {
            UNIT_ASSERT(false);
        };
        UNIT_ASSERT_TEST_FAILS(unitAssert());

        UNIT_ASSERT(true);
    }

    Y_UNIT_TEST(TypesEqual) {
        auto typesEqual = [] {
            UNIT_ASSERT_TYPES_EQUAL(int, long);
        };
        UNIT_ASSERT_TEST_FAILS(typesEqual());

        UNIT_ASSERT_TYPES_EQUAL(TString, TString);
    }

    Y_UNIT_TEST(DoublesEqual) {
        auto doublesEqual = [](double d1, double d2, double precision) {
            UNIT_ASSERT_DOUBLES_EQUAL(d1, d2, precision);
        };
        UNIT_ASSERT_TEST_FAILS(doublesEqual(0.0, 0.5, 0.1));
        UNIT_ASSERT_TEST_FAILS(doublesEqual(0.1, -0.1, 0.1));

        UNIT_ASSERT_DOUBLES_EQUAL(0.0, 0.01, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(0.01, 0.0, 0.1);

        constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
        UNIT_ASSERT_TEST_FAILS(doublesEqual(nan, 0.5, 0.1));
        UNIT_ASSERT_TEST_FAILS(doublesEqual(0.5, nan, 0.1));
        UNIT_ASSERT_DOUBLES_EQUAL(nan, nan, 0.1);
    }

    Y_UNIT_TEST(StringsEqual) {
        auto stringsEqual = [](auto s1, auto s2) {
            UNIT_ASSERT_STRINGS_EQUAL(s1, s2);
        };
        UNIT_ASSERT_TEST_FAILS(stringsEqual("q", "w"));
        UNIT_ASSERT_TEST_FAILS(stringsEqual("q", TString("w")));
        UNIT_ASSERT_TEST_FAILS(stringsEqual(TString("q"), "w"));
        UNIT_ASSERT_TEST_FAILS(stringsEqual(TString("a"), TString("b")));
        UNIT_ASSERT_TEST_FAILS(stringsEqual(TString("a"), TStringBuf("b")));
        UNIT_ASSERT_TEST_FAILS(stringsEqual("a", TStringBuf("b")));
        UNIT_ASSERT_TEST_FAILS(stringsEqual(TStringBuf("a"), "b"));

        TString empty;
        TStringBuf emptyBuf;
        UNIT_ASSERT_STRINGS_EQUAL("", empty);
        UNIT_ASSERT_STRINGS_EQUAL(empty, emptyBuf);
        UNIT_ASSERT_STRINGS_EQUAL("", static_cast<const char*>(nullptr));
    }

    Y_UNIT_TEST(StringContains) {
        auto stringContains = [](auto s, auto substr) {
            UNIT_ASSERT_STRING_CONTAINS(s, substr);
        };
        UNIT_ASSERT_TEST_FAILS(stringContains("", "a"));
        UNIT_ASSERT_TEST_FAILS(stringContains("lurkmore", "moar"));

        UNIT_ASSERT_STRING_CONTAINS("", "");
        UNIT_ASSERT_STRING_CONTAINS("a", "");
        UNIT_ASSERT_STRING_CONTAINS("failure", "fail");
        UNIT_ASSERT_STRING_CONTAINS("lurkmore", "more");
    }

    Y_UNIT_TEST(NoDiff) {
        auto noDiff = [](auto s1, auto s2) {
            UNIT_ASSERT_NO_DIFF(s1, s2);
        };
        UNIT_ASSERT_TEST_FAILS(noDiff("q", "w"));
        UNIT_ASSERT_TEST_FAILS(noDiff("q", ""));

        UNIT_ASSERT_NO_DIFF("", "");
        UNIT_ASSERT_NO_DIFF("a", "a");
    }

    Y_UNIT_TEST(StringsUnequal) {
        auto stringsUnequal = [](auto s1, auto s2) {
            UNIT_ASSERT_STRINGS_UNEQUAL(s1, s2);
        };
        UNIT_ASSERT_TEST_FAILS(stringsUnequal("1", "1"));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal("", ""));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal("42", TString("42")));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal(TString("4"), "4"));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal("d", TStringBuf("d")));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal(TStringBuf("yandex"), "yandex"));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal(TStringBuf("index"), TString("index")));
        UNIT_ASSERT_TEST_FAILS(stringsUnequal(TString("diff"), TStringBuf("diff")));

        UNIT_ASSERT_STRINGS_UNEQUAL("1", "2");
        UNIT_ASSERT_STRINGS_UNEQUAL("", "3");
        UNIT_ASSERT_STRINGS_UNEQUAL("green", TStringBuf("red"));
        UNIT_ASSERT_STRINGS_UNEQUAL(TStringBuf("solomon"), "golovan");
        UNIT_ASSERT_STRINGS_UNEQUAL("d", TString("f"));
        UNIT_ASSERT_STRINGS_UNEQUAL(TString("yandex"), "index");
        UNIT_ASSERT_STRINGS_UNEQUAL(TString("mail"), TStringBuf("yandex"));
        UNIT_ASSERT_STRINGS_UNEQUAL(TStringBuf("C++"), TString("python"));
    }

    Y_UNIT_TEST(Equal) {
        auto equal = [](auto v1, auto v2) {
            UNIT_ASSERT_EQUAL(v1, v2);
        };
        UNIT_ASSERT_TEST_FAILS(equal("1", TString("2")));
        UNIT_ASSERT_TEST_FAILS(equal(1, 2));
        UNIT_ASSERT_TEST_FAILS(equal(42ul, static_cast<unsigned short>(24)));

        UNIT_ASSERT_EQUAL("abc", TString("abc"));
        UNIT_ASSERT_EQUAL(12l, 12);
        UNIT_ASSERT_EQUAL(55, 55);
    }

    Y_UNIT_TEST(Unequal) {
        auto unequal = [](auto v1, auto v2) {
            UNIT_ASSERT_UNEQUAL(v1, v2);
        };
        UNIT_ASSERT_TEST_FAILS(unequal("x", TString("x")));
        UNIT_ASSERT_TEST_FAILS(unequal(1, 1));
        UNIT_ASSERT_TEST_FAILS(unequal(static_cast<unsigned short>(42), 42ul));

        UNIT_ASSERT_UNEQUAL("abc", TString("cba"));
        UNIT_ASSERT_UNEQUAL(12l, 10);
        UNIT_ASSERT_UNEQUAL(33, 50);
    }

    Y_UNIT_TEST(LessThan) {
        auto lt = [](auto v1, auto v2) {
            UNIT_ASSERT_LT(v1, v2);
        };

        // less than
        UNIT_ASSERT_LT(TStringBuf("1"), "2");
        UNIT_ASSERT_LT("2", TString("3"));
        UNIT_ASSERT_LT("abc", TString("azz"));
        UNIT_ASSERT_LT(2, 4);
        UNIT_ASSERT_LT(42ul, static_cast<unsigned short>(48));

        // equals
        UNIT_ASSERT_TEST_FAILS(lt(TStringBuf("2"), "2"));
        UNIT_ASSERT_TEST_FAILS(lt("2", TString("2")));
        UNIT_ASSERT_TEST_FAILS(lt("abc", TString("abc")));
        UNIT_ASSERT_TEST_FAILS(lt(2, 2));
        UNIT_ASSERT_TEST_FAILS(lt(42ul, static_cast<unsigned short>(42)));

        // greater than
        UNIT_ASSERT_TEST_FAILS(lt(TStringBuf("2"), "1"));
        UNIT_ASSERT_TEST_FAILS(lt("3", TString("2")));
        UNIT_ASSERT_TEST_FAILS(lt("azz", TString("abc")));
        UNIT_ASSERT_TEST_FAILS(lt(5, 2));
        UNIT_ASSERT_TEST_FAILS(lt(100ul, static_cast<unsigned short>(42)));
    }

    Y_UNIT_TEST(LessOrEqual) {
        auto le = [](auto v1, auto v2) {
            UNIT_ASSERT_LE(v1, v2);
        };

        // less than
        UNIT_ASSERT_LE(TStringBuf("1"), "2");
        UNIT_ASSERT_LE("2", TString("3"));
        UNIT_ASSERT_LE("abc", TString("azz"));
        UNIT_ASSERT_LE(2, 4);
        UNIT_ASSERT_LE(42ul, static_cast<unsigned short>(48));

        // equals
        UNIT_ASSERT_LE(TStringBuf("2"), "2");
        UNIT_ASSERT_LE("2", TString("2"));
        UNIT_ASSERT_LE("abc", TString("abc"));
        UNIT_ASSERT_LE(2, 2);
        UNIT_ASSERT_LE(42ul, static_cast<unsigned short>(42));

        // greater than
        UNIT_ASSERT_TEST_FAILS(le(TStringBuf("2"), "1"));
        UNIT_ASSERT_TEST_FAILS(le("3", TString("2")));
        UNIT_ASSERT_TEST_FAILS(le("azz", TString("abc")));
        UNIT_ASSERT_TEST_FAILS(le(5, 2));
        UNIT_ASSERT_TEST_FAILS(le(100ul, static_cast<unsigned short>(42)));
    }

    Y_UNIT_TEST(GreaterThan) {
        auto gt = [](auto v1, auto v2) {
            UNIT_ASSERT_GT(v1, v2);
        };

        // less than
        UNIT_ASSERT_TEST_FAILS(gt(TStringBuf("1"), "2"));
        UNIT_ASSERT_TEST_FAILS(gt("2", TString("3")));
        UNIT_ASSERT_TEST_FAILS(gt("abc", TString("azz")));
        UNIT_ASSERT_TEST_FAILS(gt(2, 4));
        UNIT_ASSERT_TEST_FAILS(gt(42ul, static_cast<unsigned short>(48)));

        // equals
        UNIT_ASSERT_TEST_FAILS(gt(TStringBuf("2"), "2"));
        UNIT_ASSERT_TEST_FAILS(gt("2", TString("2")));
        UNIT_ASSERT_TEST_FAILS(gt("abc", TString("abc")));
        UNIT_ASSERT_TEST_FAILS(gt(2, 2));
        UNIT_ASSERT_TEST_FAILS(gt(42ul, static_cast<unsigned short>(42)));

        // greater than
        UNIT_ASSERT_GT(TStringBuf("2"), "1");
        UNIT_ASSERT_GT("3", TString("2"));
        UNIT_ASSERT_GT("azz", TString("abc"));
        UNIT_ASSERT_GT(5, 2);
        UNIT_ASSERT_GT(100ul, static_cast<unsigned short>(42));
    }

    Y_UNIT_TEST(GreaterOrEqual) {
        auto ge = [](auto v1, auto v2) {
            UNIT_ASSERT_GE(v1, v2);
        };

        // less than
        UNIT_ASSERT_TEST_FAILS(ge(TStringBuf("1"), "2"));
        UNIT_ASSERT_TEST_FAILS(ge("2", TString("3")));
        UNIT_ASSERT_TEST_FAILS(ge("abc", TString("azz")));
        UNIT_ASSERT_TEST_FAILS(ge(2, 4));
        UNIT_ASSERT_TEST_FAILS(ge(42ul, static_cast<unsigned short>(48)));

        // equals
        UNIT_ASSERT_GE(TStringBuf("2"), "2");
        UNIT_ASSERT_GE("2", TString("2"));
        UNIT_ASSERT_GE("abc", TString("abc"));
        UNIT_ASSERT_GE(2, 2);
        UNIT_ASSERT_GE(42ul, static_cast<unsigned short>(42));

        // greater than
        UNIT_ASSERT_GE(TStringBuf("2"), "1");
        UNIT_ASSERT_GE("3", TString("2"));
        UNIT_ASSERT_GE("azz", TString("abc"));
        UNIT_ASSERT_GE(5, 2);
        UNIT_ASSERT_GE(100ul, static_cast<unsigned short>(42));
    }

    Y_UNIT_TEST(ValuesEqual) {
        auto valuesEqual = [](auto v1, auto v2) {
            UNIT_ASSERT_VALUES_EQUAL(v1, v2);
        };
        UNIT_ASSERT_TEST_FAILS(valuesEqual(1, 2));
        UNIT_ASSERT_TEST_FAILS(valuesEqual(1l, static_cast<short>(2)));

        UNIT_ASSERT_VALUES_EQUAL("yandex", TString("yandex"));
        UNIT_ASSERT_VALUES_EQUAL(1.0, 1.0);
    }

    Y_UNIT_TEST(ValuesUnequal) {
        auto valuesUnequal = [](auto v1, auto v2) {
            UNIT_ASSERT_VALUES_UNEQUAL(v1, v2);
        };
        UNIT_ASSERT_TEST_FAILS(valuesUnequal(5, 5));
        UNIT_ASSERT_TEST_FAILS(valuesUnequal(static_cast<char>(5), 5l));
        TString test("test");
        UNIT_ASSERT_TEST_FAILS(valuesUnequal("test", test.data()));

        UNIT_ASSERT_VALUES_UNEQUAL("UNIT_ASSERT_VALUES_UNEQUAL", "UNIT_ASSERT_VALUES_EQUAL");
        UNIT_ASSERT_VALUES_UNEQUAL(1.0, 1.1);
    }

    class TTestException: public yexception {
    public:
        TTestException(const TString& text = "test exception", bool throwMe = true)
            : ThrowMe(throwMe)
        {
            *this << text;
        }

        virtual ~TTestException() = default;

        virtual void Throw() {
            if (ThrowMe) {
                throw *this;
            }
        }

        std::string ThrowStr() {
            if (ThrowMe) {
                throw *this;
            }

            return {};
        }

        void AssertNoException() {
            UNIT_ASSERT_NO_EXCEPTION(Throw());
        }

        void AssertNoExceptionRet() {
            const TString res = UNIT_ASSERT_NO_EXCEPTION_RESULT(ThrowStr()); 
        }

        template <class TExpectedException>
        void AssertException() {
            UNIT_ASSERT_EXCEPTION(Throw(), TExpectedException);
        }

        template <class TExpectedException, class T>
        void AssertExceptionContains(const T& substr) {
            UNIT_ASSERT_EXCEPTION_CONTAINS(Throw(), TExpectedException, substr);
        }

        template <class TExpectedException, class P>
        void AssertExceptionSatisfies(const P& predicate) {
            UNIT_ASSERT_EXCEPTION_SATISFIES(Throw(), TExpectedException, predicate);
        }

        int GetValue() const {
            return 5;           // just some value for predicate testing
        }

        bool ThrowMe;
    };

    class TOtherTestException: public TTestException {
    public:
        using TTestException::TTestException;

        // Throws other type of exception
        void Throw() override {
            if (ThrowMe) {
                throw *this;
            }
        }
    };

    Y_UNIT_TEST(Exception) {
        UNIT_ASSERT_TEST_FAILS(TTestException("", false).AssertException<TTestException>());
        UNIT_ASSERT_TEST_FAILS(TTestException().AssertException<TOtherTestException>());

        UNIT_ASSERT_EXCEPTION(TOtherTestException().Throw(), TTestException);
        UNIT_ASSERT_EXCEPTION(TTestException().Throw(), TTestException);
    }

    Y_UNIT_TEST(ExceptionAssertionContainsOtherExceptionMessage) {
        NUnitTest::TUnitTestFailChecker checker;
        {
            auto guard = checker.InvokeGuard();
            TTestException("custom exception message").AssertException<TOtherTestException>();
        }
        UNIT_ASSERT(checker.Failed());
        UNIT_ASSERT_STRING_CONTAINS(checker.Msg(), "custom exception message");
    }

    Y_UNIT_TEST(NoException) {
        UNIT_ASSERT_TEST_FAILS(TTestException().AssertNoException());
        UNIT_ASSERT_TEST_FAILS(TTestException().AssertNoExceptionRet());

        UNIT_ASSERT_NO_EXCEPTION(TTestException("", false).Throw());
    }

    Y_UNIT_TEST(ExceptionContains) {
        UNIT_ASSERT_TEST_FAILS(TTestException("abc").AssertExceptionContains<TTestException>("cba"));
        UNIT_ASSERT_TEST_FAILS(TTestException("abc").AssertExceptionContains<TTestException>(TStringBuf("cba")));
        UNIT_ASSERT_TEST_FAILS(TTestException("abc").AssertExceptionContains<TTestException>(TString("cba")));
        UNIT_ASSERT_TEST_FAILS(TTestException("abc").AssertExceptionContains<TTestException>(TStringBuilder() << "cba"));

        UNIT_ASSERT_TEST_FAILS(TTestException("abc", false).AssertExceptionContains<TTestException>("bc"));

        UNIT_ASSERT_TEST_FAILS(TTestException("abc").AssertExceptionContains<TOtherTestException>("b"));

        UNIT_ASSERT_EXCEPTION_CONTAINS(TTestException("abc").Throw(), TTestException, "a");
    }

    Y_UNIT_TEST(ExceptionSatisfies) {
        const auto goodPredicate = [](const TTestException& e) { return e.GetValue() == 5; };
        const auto badPredicate  = [](const TTestException& e) { return e.GetValue() != 5; };
        UNIT_ASSERT_NO_EXCEPTION(TTestException().AssertExceptionSatisfies<TTestException>(goodPredicate));
        UNIT_ASSERT_TEST_FAILS(TTestException().AssertExceptionSatisfies<TTestException>(badPredicate));
        UNIT_ASSERT_TEST_FAILS(TTestException().AssertExceptionSatisfies<TOtherTestException>(goodPredicate));
    }
}
