#include "scope.h"

#include <util/generic/ptr.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(ScopeToolsTest) {
    Y_UNIT_TEST(OnScopeExitTest) {
        int i = 0;

        {
            Y_SCOPE_EXIT(&i) {
                i = i * 2;
            };

            Y_SCOPE_EXIT(&i) {
                i = i + 1;
            };
        }

        UNIT_ASSERT_VALUES_EQUAL(2, i);
    }

    Y_UNIT_TEST(OnScopeExitMoveTest) {
        THolder<int> i{new int{10}};
        int p = 0;

        {
            Y_SCOPE_EXIT(i = std::move(i), &p) {
                p = *i * 2;
            };
        }

        UNIT_ASSERT_VALUES_EQUAL(20, p);
    }

    Y_UNIT_TEST(TestDefer) {
        int i = 0;

        {
            Y_DEFER {
                i = 20;
            };
        }
        UNIT_ASSERT_VALUES_EQUAL(i, 20);
    }

    Y_UNIT_TEST(TestDeferred) {
        int i = 0;

        {
            TDeferredOnceFunction writeI([&]() noexcept { i = 1; });
            {
                auto doWriteI = std::move(writeI);
                UNIT_ASSERT_VALUES_EQUAL(i, 0);
            } // doWriteI called
            UNIT_ASSERT_VALUES_EQUAL(i, 1);

            TDeferredOnceFunction updateI([&]() noexcept { i = 2; });
            UNIT_ASSERT_VALUES_EQUAL(i, 1);
            std::move(updateI).CallNow();
            UNIT_ASSERT_VALUES_EQUAL(i, 2);
        } // moved-from writeI is no-op
        UNIT_ASSERT_VALUES_EQUAL(i, 2);

        TDeferredOnceFunction droppedUpdateI([&]() noexcept { i = 3; });
        std::move(droppedUpdateI).Drop();
        UNIT_ASSERT_VALUES_EQUAL(i, 2);
    }
} // Y_UNIT_TEST_SUITE(ScopeToolsTest)
