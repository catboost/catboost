#include "explicit_type.h"

#include <library/cpp/testing/unittest/registar.h>

struct TCallableBase {
public:
    using TYes = char;
    using TNo = struct {
        TYes dummy[32];
    };

    template <class T, class Arg>
    static TNo Test(const T&, const Arg&, ...);

    template <class T, class Arg>
    static TYes Test(const T&, const Arg&, int, decltype(std::declval<T>()(std::declval<Arg>()))* = nullptr);
};

template <class T, class Arg>
struct TCallable: public TCallableBase {
    enum {
        Result = sizeof(Test(std::declval<T>(), std::declval<Arg>(), 1)) == sizeof(TYes)
    };
};

template <class T>
struct TExplicitlyCallable {
    void operator()(TExplicitType<T>) {
    }
};

struct IntConvertible {
    operator int() {
        return 1;
    }
};

struct IntConstructible {
    IntConstructible(const int&) {
    }
};

Y_UNIT_TEST_SUITE(TestExplicitType) {
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<char>, char>::Result), true);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<char>, int>::Result), false);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<char>, wchar_t>::Result), false);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<int>, int>::Result), true);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<int>, IntConvertible>::Result), false);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<IntConstructible>, IntConstructible>::Result), true);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<IntConstructible>, IntConvertible>::Result), false);
        UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(TCallable<TExplicitlyCallable<IntConstructible>, int>::Result), false);
    }
}
