#include <library/cpp/testing/unittest/registar.h>

#include "flags.h"

enum ETestFlag1: ui16 {
    Test1 = 1,
    Test2 = 2,
    Test4 = 4,
    Test8 = 8
};
Y_DECLARE_FLAGS(ETest1, ETestFlag1);
Y_DECLARE_OPERATORS_FOR_FLAGS(ETest1);

static_assert(TTypeTraits<ETest1>::IsPod, "flags should be POD type");

enum class ETestFlag2 {
    Test1 = 1,
    Test2 = 2,
    Test4 = 4,
    Test8 = 8
};
Y_DECLARE_FLAGS(ETest2, ETestFlag2);
Y_DECLARE_OPERATORS_FOR_FLAGS(ETest2);

namespace {
    // won't compile without Y_DECLARE_UNUSED
    enum class ETestFlag3 { One = 1,
                            Two = 2,
                            Three = 3 };
    Y_DECLARE_FLAGS(ETestFlags3, ETestFlag3);
    Y_DECLARE_OPERATORS_FOR_FLAGS(ETestFlags3);
} // namespace

Y_UNIT_TEST_SUITE(TFlagsTest) {
    template <class Enum>
    void TestEnum() {
        {
            auto i = Enum::Test1 | Enum::Test2;

            UNIT_ASSERT((std::is_same<decltype(i), TFlags<Enum>>::value));
            UNIT_ASSERT((std::is_same<decltype(~i), TFlags<Enum>>::value));
            UNIT_ASSERT(!(std::is_same<decltype(i), int>::value));
            UNIT_ASSERT_VALUES_EQUAL(sizeof(Enum), sizeof(TFlags<Enum>));

            UNIT_ASSERT(i.HasFlag(Enum::Test1));
            UNIT_ASSERT(i.HasFlag(Enum::Test4) == false);
            UNIT_ASSERT(i.HasFlags(Enum::Test1));
            UNIT_ASSERT(i.HasFlags(Enum::Test4) == false);
            UNIT_ASSERT(i.HasFlags(Enum::Test1 | Enum::Test4) == false);
            UNIT_ASSERT(i.HasAnyOfFlags(Enum::Test1));
            UNIT_ASSERT(i.HasAnyOfFlags(Enum::Test4) == false);
            UNIT_ASSERT(i.HasAnyOfFlags(Enum::Test1 | Enum::Test4));

            i |= Enum::Test4;
            i ^= Enum::Test2;
            UNIT_ASSERT_EQUAL(i, Enum::Test4 | Enum::Test1);
            UNIT_ASSERT_EQUAL(i & Enum::Test1, i & ~Enum::Test4);
            UNIT_ASSERT(i & Enum::Test4);
            UNIT_ASSERT_UNEQUAL(i, ~i);
            UNIT_ASSERT_EQUAL(i, ~~i);
        }
        {
            auto i = Enum::Test1 | Enum::Test2;
            i.RemoveFlag(Enum::Test1);
            UNIT_ASSERT_EQUAL(i, TFlags<Enum>(Enum::Test2));
        }
        {
            auto i = Enum::Test1 | Enum::Test2;
            i.RemoveFlags(Enum::Test1);
            UNIT_ASSERT_EQUAL(i, TFlags<Enum>(Enum::Test2));
        }
        {
            auto i = Enum::Test1 | Enum::Test2;
            i.RemoveFlags(Enum::Test1 | Enum::Test2);
            UNIT_ASSERT_EQUAL(i, TFlags<Enum>());
        }
    }

    Y_UNIT_TEST(TestFlags) {
        TestEnum<ETestFlag1>();
        TestEnum<ETestFlag2>();
    }

    Y_UNIT_TEST(TestZero) {
        /*  This code should simply compile. */

        ETest1 f = 0;
        f = 0;
        f = ETest1(0);

        ETest1 ff(0);
        ff = 0;
    }

    Y_UNIT_TEST(TestOutput) {
        ETest1 value0 = nullptr, value1 = Test1, value7 = Test1 | Test2 | Test4;

        UNIT_ASSERT_VALUES_EQUAL(ToString(value0), "TFlags(0000000000000000)");
        UNIT_ASSERT_VALUES_EQUAL(ToString(value1), "TFlags(0000000000000001)");
        UNIT_ASSERT_VALUES_EQUAL(ToString(value7), "TFlags(0000000000000111)");
    }

    Y_UNIT_TEST(TestHash) {
        ETest1 value0 = nullptr, value1 = Test1;

        THashMap<ETest1, int> hash;
        hash[value0] = 0;
        hash[value1] = 1;

        UNIT_ASSERT_VALUES_EQUAL(hash[value0], 0);
        UNIT_ASSERT_VALUES_EQUAL(hash[value1], 1);
    }

    Y_UNIT_TEST(TestBaseType) {
        ui16 goodValue = 7;
        auto goodFlags = ETest1::FromBaseType(goodValue);
        UNIT_ASSERT(goodFlags & ETestFlag1::Test1);
        UNIT_ASSERT(goodFlags & ETestFlag1::Test2);
        UNIT_ASSERT(goodFlags & ETestFlag1::Test4);
        UNIT_ASSERT_VALUES_EQUAL(goodValue, goodFlags.ToBaseType());

        // Passed value is not checked, but preserved as is
        ui16 badValue = 1024;
        auto badFlags = ETest1::FromBaseType(badValue);
        UNIT_ASSERT_VALUES_EQUAL(badValue, badFlags.ToBaseType());
    }
} // Y_UNIT_TEST_SUITE(TFlagsTest)
