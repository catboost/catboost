#include "lazy_value.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TLazyValueTestSuite) {
    SIMPLE_UNIT_TEST(TestLazyValue) {
        TLazyValue<int> value([]() {
            return 5;
        });
        UNIT_ASSERT(!value);
        UNIT_ASSERT_EQUAL(*value, 5);
        UNIT_ASSERT(value);
    }

    class TValueProvider {
        TLazyValue<TString> Data;

        TString ParseData() {
            return "hi";
        }

    public:
        TValueProvider()
            : Data([&] { return this->ParseData(); })
        {
        }

        const TString& GetData() const {
            return *Data;
        }
    };

    SIMPLE_UNIT_TEST(TestValueProvider) {
        TValueProvider provider;

        UNIT_ASSERT(provider.GetData() == "hi");
    }

    SIMPLE_UNIT_TEST(TestMakeLazy) {
        auto lv = MakeLazy([] {
            return 100500;
        });
        UNIT_ASSERT(!lv);
        UNIT_ASSERT(lv.GetRef() == 100500);
        UNIT_ASSERT(lv);
    }
}
