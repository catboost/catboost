#include "singleton.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestSingleton) {
    struct THuge {
        char Buf[1000000];
        int V = 1234;
    };

    Y_UNIT_TEST(TestHuge) {
        UNIT_ASSERT_VALUES_EQUAL(*HugeSingleton<int>(), 0);
        UNIT_ASSERT_VALUES_EQUAL(HugeSingleton<THuge>()->V, 1234);
    }

    struct TWithParams {
        explicit TWithParams(const ui32 data1 = 0, const TString& data2 = TString())
            : Data1(data1)
            , Data2(data2)
        {
        }

        ui32 Data1;
        TString Data2;
    };

    Y_UNIT_TEST(TestConstructorParamsOrder) {
        UNIT_ASSERT_VALUES_EQUAL(Singleton<TWithParams>(10, "123123")->Data1, 10);
        UNIT_ASSERT_VALUES_EQUAL(Singleton<TWithParams>(20, "123123")->Data1, 10);
        UNIT_ASSERT_VALUES_EQUAL(Singleton<TWithParams>(10, "456456")->Data2, "123123");
    }

    Y_UNIT_TEST(TestInstantiationWithConstructorParams) {
        UNIT_ASSERT_VALUES_EQUAL(Singleton<TWithParams>(10)->Data1, 10);
        UNIT_ASSERT_VALUES_EQUAL(HugeSingleton<TWithParams>(20, "123123")->Data2, "123123");
        {
            const auto value = SingletonWithPriority<TWithParams, 12312>(30, "456")->Data1;
            UNIT_ASSERT_VALUES_EQUAL(value, 30);
        }
        {
            const auto value = HugeSingletonWithPriority<TWithParams, 12311>(40, "789")->Data2;
            UNIT_ASSERT_VALUES_EQUAL(value, "789");
        }
        UNIT_ASSERT_VALUES_EQUAL(Default<TWithParams>().Data1, 0);
    }
} // Y_UNIT_TEST_SUITE(TestSingleton)
