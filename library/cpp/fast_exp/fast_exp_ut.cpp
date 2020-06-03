#include "fast_exp.h"

#include <library/cpp/unittest/registar.h>

#include <cmath>

Y_UNIT_TEST_SUITE(TestFastExp) {
    Y_UNIT_TEST(TestStd) {
        for (size_t i = 1; i < 10; ++i) {
            UNIT_ASSERT(fabs(fast_exp(i) - std::exp(i)) < 0.01);
        }
    }
    Y_UNIT_TEST(TestVector) {
        constexpr int count = 5;
        double tmp[count] = {0, 1, 2, 3, 4};
        FastExpInplace(tmp, count);
        for (size_t i = 0; i < count; ++i) {
            UNIT_ASSERT(tmp[i] > 0 && fabs(1 - std::exp(i) / tmp[i]) < 0.01);
        }
        FastExpInplace(tmp + 1, count - 1);
        for (size_t i = 1; i < count; ++i) {
            UNIT_ASSERT(tmp[i] > 0 && fabs(1 - std::exp(std::exp(i)) / tmp[i]) < 0.01);
        }
    }
}
