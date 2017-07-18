#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/str.h>
#include <library/unittest/registar.h>

#include "maybe.h"

class TIncrementOnDestroy {
private:
    int* Ptr;

public:
    TIncrementOnDestroy(int* ptr) noexcept
        : Ptr(ptr)
    {
    }

    ~TIncrementOnDestroy() {
        ++*Ptr;
    }
};

SIMPLE_UNIT_TEST_SUITE(TMaybeTest) {
    SIMPLE_UNIT_TEST(TestWarning) {
        TMaybe<size_t> x;
        TStringStream ss;
        TString line;

        while (ss.ReadLine(line)) {
            x = line.size();
        }

        if (x == 5u) {
            ss << "5\n";
        }
    }

    SIMPLE_UNIT_TEST(TTestConstructorDestructor) {
        int a = 0;
        int b = 0;

        TMaybe<TIncrementOnDestroy>();
        UNIT_ASSERT_VALUES_EQUAL(a, b);

        TMaybe<TIncrementOnDestroy>(TIncrementOnDestroy(&a));
        b += 2;
        UNIT_ASSERT_VALUES_EQUAL(a, b);

        {
            TMaybe<TIncrementOnDestroy> m1 = TIncrementOnDestroy(&a);
            b += 1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);

            TMaybe<TIncrementOnDestroy> m2 = m1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);

            TMaybe<TIncrementOnDestroy> m3;
            m3 = m1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);
        }

        b += 3;
        UNIT_ASSERT_VALUES_EQUAL(a, b);

        {
            TMaybe<TIncrementOnDestroy> m4 = TIncrementOnDestroy(&a);
            b += 1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);

            m4 = TIncrementOnDestroy(&a);
            b += 1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);

            m4.Clear();
            b += 1;
            UNIT_ASSERT_VALUES_EQUAL(a, b);

            m4.Clear();
            UNIT_ASSERT_VALUES_EQUAL(a, b);
        }
    }

    SIMPLE_UNIT_TEST(TestAssignmentClear) {
        TMaybe<int> m5;
        UNIT_ASSERT(!m5.Defined());
        UNIT_ASSERT(m5.Empty());
        UNIT_ASSERT(m5 == TMaybe<int>());
        UNIT_ASSERT(m5 == Nothing());
        UNIT_ASSERT(m5 != TMaybe<int>(4));

        m5 = 4;

        UNIT_ASSERT(m5.Defined());
        UNIT_ASSERT(!m5.Empty());

        UNIT_ASSERT_VALUES_EQUAL(4, m5.GetRef());
        UNIT_ASSERT(m5 == TMaybe<int>(4));
        UNIT_ASSERT(m5 != TMaybe<int>(3));
        UNIT_ASSERT(m5 != TMaybe<int>());
        UNIT_ASSERT(m5 != Nothing());

        m5 = TMaybe<int>(5);
        UNIT_ASSERT(m5.Defined());
        UNIT_ASSERT_VALUES_EQUAL(5, m5.GetRef());
        UNIT_ASSERT(m5 == TMaybe<int>(5));
        UNIT_ASSERT(m5 != TMaybe<int>(4));

        m5 = TMaybe<int>();
        UNIT_ASSERT(m5.Empty());
        UNIT_ASSERT(m5 == TMaybe<int>());
        UNIT_ASSERT(m5 == Nothing());
        UNIT_ASSERT(m5 != TMaybe<int>(5));

        m5 = 4;
        m5 = Nothing();

        UNIT_ASSERT(m5.Empty());
        UNIT_ASSERT(m5 == TMaybe<int>());
        UNIT_ASSERT(m5 == Nothing());
        UNIT_ASSERT(m5 != TMaybe<int>(5));
    }

    SIMPLE_UNIT_TEST(TestInPlace) {
        TMaybe<int> m;

        UNIT_ASSERT(!m);

        m.ConstructInPlace(1);

        UNIT_ASSERT(m == 1);

        auto& x = m.ConstructInPlace(2);

        UNIT_ASSERT(m == 2);
        x = 7;
        UNIT_ASSERT(m == 7);
    }

    SIMPLE_UNIT_TEST(TestMove) {
        struct TMovable {
            int Flag = 0;

            TMovable(int flag)
                : Flag(flag)
            {
            }

            TMovable(const TMovable&) = delete;
            TMovable& operator=(const TMovable&) = delete;

            TMovable(TMovable&& other) {
                std::swap(Flag, other.Flag);
            }
            TMovable& operator=(TMovable&& other) {
                std::swap(Flag, other.Flag);
                return *this;
            }
        };

        // Move ctor from value
        TMovable value1(1);
        TMaybe<TMovable> m1(std::move(value1));
        UNIT_ASSERT(m1.Defined());
        UNIT_ASSERT_VALUES_EQUAL(m1->Flag, 1);

        // Move assignment from value
        TMovable value2(2);
        TMaybe<TMovable> m2;
        m2 = std::move(value2);
        UNIT_ASSERT(m2.Defined());
        UNIT_ASSERT_VALUES_EQUAL(m2->Flag, 2);

        // Move ctor from maybe
        TMaybe<TMovable> m3(std::move(m1));
        UNIT_ASSERT(m3.Defined());
        UNIT_ASSERT_VALUES_EQUAL(m3->Flag, 1);

        // Move assignment from maybe
        TMaybe<TMovable> m4;
        m4 = std::move(m2);
        UNIT_ASSERT(m4.Defined());
        UNIT_ASSERT_VALUES_EQUAL(m4->Flag, 2);
    }

    SIMPLE_UNIT_TEST(TestCast) {
        // Undefined maybe casts to undefined maybe
        TMaybe<short> shortMaybe;
        const auto undefinedMaybe = shortMaybe.Cast<long>();
        UNIT_ASSERT(!undefinedMaybe.Defined());

        // Defined maybe casts to defined maybe of another type
        shortMaybe = 34;
        const auto longMaybe = shortMaybe.Cast<long>();
        UNIT_ASSERT(longMaybe.Defined());
        UNIT_ASSERT_VALUES_EQUAL(34, longMaybe.GetRef());
    }

    SIMPLE_UNIT_TEST(TestGetOr) {
        UNIT_ASSERT_VALUES_EQUAL(TMaybe<TString>().GetOrElse("xxx"), TString("xxx"));
        UNIT_ASSERT_VALUES_EQUAL(TMaybe<TString>("yyy").GetOrElse("xxx"), TString("yyy"));

        {
            TString xxx = "xxx";
            UNIT_ASSERT_VALUES_EQUAL(TMaybe<TString>().GetOrElse(xxx).append('x'), TString("xxxx"));
            UNIT_ASSERT_VALUES_EQUAL(xxx, "xxxx");
        }

        {
            TString xxx = "xxx";
            UNIT_ASSERT_VALUES_EQUAL(TMaybe<TString>("yyy").GetOrElse(xxx).append('x'), TString("yyyx"));
            UNIT_ASSERT_VALUES_EQUAL(xxx, "xxx");
        }
    }

    /*
  ==
  !=
  <
  <=
  >
  >=
*/

    SIMPLE_UNIT_TEST(TestCompareEqualEmpty) {
        TMaybe<int> m1;
        TMaybe<int> m2;

        UNIT_ASSERT(m1 == m2);
        UNIT_ASSERT(!(m1 != m2));
        UNIT_ASSERT(!(m1 < m2));
        UNIT_ASSERT(m1 <= m2);
        UNIT_ASSERT(!(m1 > m2));
        UNIT_ASSERT(m1 >= m2);
    }

    SIMPLE_UNIT_TEST(TestCompareEqualNonEmpty) {
        TMaybe<int> m1{1};
        TMaybe<int> m2{1};

        UNIT_ASSERT(m1 == m2);
        UNIT_ASSERT(!(m1 != m2));
        UNIT_ASSERT(!(m1 < m2));
        UNIT_ASSERT(m1 <= m2);
        UNIT_ASSERT(!(m1 > m2));
        UNIT_ASSERT(m1 >= m2);
    }

    SIMPLE_UNIT_TEST(TestCompareOneLessThanOther) {
        TMaybe<int> m1{1};
        TMaybe<int> m2{2};

        UNIT_ASSERT(!(m1 == m2));
        UNIT_ASSERT(m1 != m2);
        UNIT_ASSERT(m1 < m2);
        UNIT_ASSERT(m1 <= m2);
        UNIT_ASSERT(!(m1 > m2));
        UNIT_ASSERT(!(m1 >= m2));
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndT_Equal) {
        TMaybe<int> m{1};
        int v{1};

        UNIT_ASSERT(m == v);
        UNIT_ASSERT(!(m != v));
        UNIT_ASSERT(!(m < v));
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(m >= v);

        UNIT_ASSERT(v == m);
        UNIT_ASSERT(!(v != m));
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(v <= m);
        UNIT_ASSERT(!(v > m));
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndT_TMaybeLessThanT) {
        TMaybe<int> m{1};
        int v{2};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(m < v);
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(!(m >= v));

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(!(v <= m));
        UNIT_ASSERT(v > m);
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndT_TMaybeGreaterThanT) {
        TMaybe<int> m{2};
        int v{1};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(!(m < v));
        UNIT_ASSERT(!(m <= v));
        UNIT_ASSERT(m > v);
        UNIT_ASSERT(m >= v);

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(v < m);
        UNIT_ASSERT(v <= m);
        UNIT_ASSERT(!(v > m));
        UNIT_ASSERT(!(v >= m));
    }

    SIMPLE_UNIT_TEST(TestCompareEmptyTMaybeAndT) {
        TMaybe<int> m;
        int v{1};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(m < v);
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(!(m >= v));

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(!(v <= m));
        UNIT_ASSERT(v > m);
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareEmptyTMaybeAndNothing) {
        TMaybe<int> m;
        auto n = Nothing();

        UNIT_ASSERT(m == n);
        UNIT_ASSERT(!(m != n));
        UNIT_ASSERT(!(m < n));
        UNIT_ASSERT(m <= n);
        UNIT_ASSERT(!(m > n));
        UNIT_ASSERT(m >= n);

        UNIT_ASSERT(n == m);
        UNIT_ASSERT(!(n != m));
        UNIT_ASSERT(!(n < m));
        UNIT_ASSERT(n <= m);
        UNIT_ASSERT(!(n > m));
        UNIT_ASSERT(n >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareNonEmptyTMaybeAndNothing) {
        TMaybe<int> m{1};
        auto n = Nothing();

        UNIT_ASSERT(!(m == n));
        UNIT_ASSERT(m != n);
        UNIT_ASSERT(!(m < n));
        UNIT_ASSERT(!(m <= n));
        UNIT_ASSERT(m > n);
        UNIT_ASSERT(m >= n);

        UNIT_ASSERT(!(n == m));
        UNIT_ASSERT(n != m);
        UNIT_ASSERT(n < m);
        UNIT_ASSERT(n <= m);
        UNIT_ASSERT(!(n > m));
        UNIT_ASSERT(!(n >= m));
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndConvertibleT_Equal) {
        TMaybe<size_t> m{1};
        unsigned int v{1};

        UNIT_ASSERT(m == v);
        UNIT_ASSERT(!(m != v));
        UNIT_ASSERT(!(m < v));
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(m >= v);

        UNIT_ASSERT(v == m);
        UNIT_ASSERT(!(v != m));
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(v <= m);
        UNIT_ASSERT(!(v > m));
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndConvertibleT_TMaybeLessThanT) {
        TMaybe<size_t> m{1};
        unsigned int v{2};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(m < v);
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(!(m >= v));

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(!(v <= m));
        UNIT_ASSERT(v > m);
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestCompareTMaybeAndConvertibleT_TMaybeGreaterThanT) {
        TMaybe<size_t> m{2};
        unsigned int v{1};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(!(m < v));
        UNIT_ASSERT(!(m <= v));
        UNIT_ASSERT(m > v);
        UNIT_ASSERT(m >= v);

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(v < m);
        UNIT_ASSERT(v <= m);
        UNIT_ASSERT(!(v > m));
        UNIT_ASSERT(!(v >= m));
    }

    SIMPLE_UNIT_TEST(TestCompareEmptyTMaybeAndConvertibleT) {
        TMaybe<size_t> m;
        unsigned int v{1};

        UNIT_ASSERT(!(m == v));
        UNIT_ASSERT(m != v);
        UNIT_ASSERT(m < v);
        UNIT_ASSERT(m <= v);
        UNIT_ASSERT(!(m > v));
        UNIT_ASSERT(!(m >= v));

        UNIT_ASSERT(!(v == m));
        UNIT_ASSERT(v != m);
        UNIT_ASSERT(!(v < m));
        UNIT_ASSERT(!(v <= m));
        UNIT_ASSERT(v > m);
        UNIT_ASSERT(v >= m);
    }

    SIMPLE_UNIT_TEST(TestMakeMaybe) {
        {
            auto m1 = MakeMaybe<int>(1);
            UNIT_ASSERT(*m1 == 1);
        }

        {
            struct TMockClass {
                TMockClass(int i)
                    : I_(i)
                {
                }

                TMockClass(const TMockClass& other)
                    : I_(other.I_)
                {
                    IsCopyConstructorCalled_ = true;
                }

                TMockClass& operator=(const TMockClass& other) {
                    if (this != &other) {
                        I_ = other.I_;
                        IsCopyAssignmentOperatorCalled_ = true;
                    }

                    return *this;
                }

                TMockClass(TMockClass&& other)
                    : I_(other.I_)
                {
                    IsMoveConstructorCalled_ = true;
                }

                TMockClass& operator=(TMockClass&& other) {
                    if (this != &other) {
                        I_ = other.I_;
                        IsMoveAssignmentOperatorCalled_ = true;
                    }

                    return *this;
                }

                int I_;
                bool IsCopyConstructorCalled_{false};
                bool IsMoveConstructorCalled_{false};
                bool IsCopyAssignmentOperatorCalled_{false};
                bool IsMoveAssignmentOperatorCalled_{false};
            };

            auto m2 = MakeMaybe<TMockClass>(1);
            UNIT_ASSERT(m2->I_ == 1);
            UNIT_ASSERT(!m2->IsCopyConstructorCalled_);
            UNIT_ASSERT(!m2->IsMoveConstructorCalled_);
            UNIT_ASSERT(!m2->IsCopyAssignmentOperatorCalled_);
            UNIT_ASSERT(!m2->IsMoveAssignmentOperatorCalled_);
        }

        {
            auto m3 = MakeMaybe<yvector<int>>({1, 2, 3, 4, 5});
            UNIT_ASSERT(m3->size() == 5);
            UNIT_ASSERT(m3->at(0) == 1);
            UNIT_ASSERT(m3->at(1) == 2);
            UNIT_ASSERT(m3->at(2) == 3);
            UNIT_ASSERT(m3->at(3) == 4);
            UNIT_ASSERT(m3->at(4) == 5);
        }

        {
            struct TMockStruct4 {
                TMockStruct4(int a, int b, int c)
                    : A_(a)
                    , B_(b)
                    , C_(c)
                {
                }

                int A_;
                int B_;
                int C_;
            };

            auto m4 = MakeMaybe<TMockStruct4>(1, 2, 3);
            UNIT_ASSERT(m4->A_ == 1);
            UNIT_ASSERT(m4->B_ == 2);
            UNIT_ASSERT(m4->C_ == 3);
        }

        {
            struct TMockStruct5 {
                TMockStruct5(const yvector<int>& vec, bool someFlag)
                    : Vec_(vec)
                    , SomeFlag_(someFlag)
                {
                }

                yvector<int> Vec_;
                bool SomeFlag_;
            };

            auto m5 = MakeMaybe<TMockStruct5>({1, 2, 3}, true);
            UNIT_ASSERT(m5->Vec_.size() == 3);
            UNIT_ASSERT(m5->Vec_[0] == 1);
            UNIT_ASSERT(m5->Vec_[1] == 2);
            UNIT_ASSERT(m5->Vec_[2] == 3);
            UNIT_ASSERT(m5->SomeFlag_);
        }
    }

    SIMPLE_UNIT_TEST(TestSwappingUsingMemberSwap) {
        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = 2;

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(*m2 == 2);

            m1.Swap(m2);

            UNIT_ASSERT(*m1 == 2);
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = Nothing();

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());

            m1.Swap(m2);

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = Nothing();
            TMaybe<int> m2 = 1;

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);

            m1.Swap(m2);

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());
        }
    }

    SIMPLE_UNIT_TEST(TestSwappingUsingMemberLittleSwap) {
        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = 2;

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(*m2 == 2);

            m1.swap(m2);

            UNIT_ASSERT(*m1 == 2);
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = Nothing();

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());

            m1.swap(m2);

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = Nothing();
            TMaybe<int> m2 = 1;

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);

            m1.swap(m2);

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());
        }
    }

    SIMPLE_UNIT_TEST(TestSwappingUsingGlobalSwap) {
        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = 2;

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(*m2 == 2);

            ::Swap(m1, m2);

            UNIT_ASSERT(*m1 == 2);
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = Nothing();

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());

            ::Swap(m1, m2);

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = Nothing();
            TMaybe<int> m2 = 1;

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);

            ::Swap(m1, m2);

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());
        }
    }

    SIMPLE_UNIT_TEST(TestSwappingUsingGlobalDoSwap) {
        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = 2;

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(*m2 == 2);

            ::DoSwap(m1, m2);

            UNIT_ASSERT(*m1 == 2);
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = Nothing();

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());

            ::DoSwap(m1, m2);

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = Nothing();
            TMaybe<int> m2 = 1;

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);

            ::DoSwap(m1, m2);

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());
        }
    }

    SIMPLE_UNIT_TEST(TestSwappingUsingStdSwap) {
        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = 2;

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(*m2 == 2);

            ::std::swap(m1, m2);

            UNIT_ASSERT(*m1 == 2);
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = 1;
            TMaybe<int> m2 = Nothing();

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());

            ::std::swap(m1, m2);

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);
        }

        {
            TMaybe<int> m1 = Nothing();
            TMaybe<int> m2 = 1;

            UNIT_ASSERT(m1 == Nothing());
            UNIT_ASSERT(*m2 == 1);

            ::std::swap(m1, m2);

            UNIT_ASSERT(*m1 == 1);
            UNIT_ASSERT(m2 == Nothing());
        }
    }

    SIMPLE_UNIT_TEST(TestOutputStreamEmptyMaybe) {
        TString s;
        TStringOutput output(s);
        output << TMaybe<int>();
        UNIT_ASSERT_EQUAL("(empty maybe)", s);
    }

    SIMPLE_UNIT_TEST(TestOutputStreamDefinedMaybe) {
        TString s;
        TStringOutput output(s);
        output << TMaybe<int>(42);
        UNIT_ASSERT_EQUAL("42", s);
    }
}
