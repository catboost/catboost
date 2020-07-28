#include <library/cpp/containers/flat_hash/lib/size_fitters.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NFlatHash;

Y_UNIT_TEST_SUITE(TAndSizeFitterTest) {
    Y_UNIT_TEST(EvalSizeTest) {
        TAndSizeFitter sf;
        UNIT_ASSERT_EQUAL(sf.EvalSize(5), 8);
        UNIT_ASSERT_EQUAL(sf.EvalSize(8), 8);
        UNIT_ASSERT_EQUAL(sf.EvalSize(13), 16);
        UNIT_ASSERT_EQUAL(sf.EvalSize(25), 32);
        for (size_t i = 1; i < 100; ++i) {
            UNIT_ASSERT_EQUAL(sf.EvalSize(i), FastClp2(i));
        }
    }

    Y_UNIT_TEST(EvalIndexTest) {
        TAndSizeFitter sf;
        for (size_t j = 1; j < 10; ++j) {
            sf.Update(1 << j);
            for (size_t i = 0; i < 100; ++i) {
                UNIT_ASSERT_EQUAL(sf.EvalIndex(i, 1 << j), i & ((1 << j) - 1));
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TModSizeFitterTest) {
    Y_UNIT_TEST(EvalSizeTest) {
        TModSizeFitter sf;
        UNIT_ASSERT_EQUAL(sf.EvalSize(5), 5);
        UNIT_ASSERT_EQUAL(sf.EvalSize(8), 8);
        UNIT_ASSERT_EQUAL(sf.EvalSize(13), 13);
        UNIT_ASSERT_EQUAL(sf.EvalSize(25), 25);
        for (size_t i = 1; i < 100; ++i) {
            UNIT_ASSERT_EQUAL(sf.EvalSize(i), i);
        }
    }

    Y_UNIT_TEST(EvalIndexTest) {
        TModSizeFitter sf;
        for (size_t j = 1; j < 10; ++j) {
            sf.Update(1 << j); // just for integrity
            for (size_t i = 0; i < 100; ++i) {
                UNIT_ASSERT_EQUAL(sf.EvalIndex(i, 1 << j), i % (1 << j));
            }
        }
    }
}
