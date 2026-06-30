#include "common_ops.h"
#include "random.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/digest/numeric.h>

#include <random>

Y_UNIT_TEST_SUITE(TestCommonRNG) {
    template <class T>
    struct TRng: public TCommonRNG<T, TRng<T>> {
        inline T GenRand() noexcept {
            return IntHash(C_++);
        }

        T C_ = RandomNumber<T>();
    };

    Y_UNIT_TEST(TestUniform1) {
        TRng<ui32> r;

        for (size_t i = 0; i < 1000; ++i) {
            UNIT_ASSERT(r.Uniform(10) < 10);
        }
    }

    Y_UNIT_TEST(TestUniform2) {
        TRng<ui32> r;

        for (size_t i = 0; i < 1000; ++i) {
            UNIT_ASSERT(r.Uniform(1) == 0);
        }
    }

    Y_UNIT_TEST(TestUniform3) {
        TRng<ui32> r;

        for (size_t i = 0; i < 1000; ++i) {
            auto x = r.Uniform(100, 200);

            UNIT_ASSERT(x >= 100);
            UNIT_ASSERT(x < 200);
        }
    }

    Y_UNIT_TEST(TestStlCompatibility) {
        {
            TRng<ui32> r;
            r.C_ = 17;
            std::normal_distribution<float> nd(0, 1);
            UNIT_ASSERT_DOUBLES_EQUAL(nd(r), -0.877167, 0.01);
        }

        {
            TRng<ui64> r;
            r.C_ = 17;
            std::normal_distribution<double> nd(0, 1);
            UNIT_ASSERT_DOUBLES_EQUAL(nd(r), -0.5615566731, 0.01);
        }

        {
            TRng<ui16> r;
            r.C_ = 17;
            std::normal_distribution<long double> nd(0, 1);
            UNIT_ASSERT_DOUBLES_EQUAL(nd(r), -0.430375088, 0.01);
        }
    }
} // Y_UNIT_TEST_SUITE(TestCommonRNG)
