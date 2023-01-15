#include <library/cpp/testing/unittest/gtest.h>
#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/generic/set.h>
#include <util/network/sock.h>
#include <util/system/env.h>
#include <util/system/fs.h>

TEST(GTest, Test1) {
    UNIT_ASSERT_EQUAL(1, 1);
}

TEST(GTest, Test2) {
    UNIT_ASSERT_EQUAL(2, 2);
}

namespace {
    struct TFixture : ::testing::Test {
        TFixture()
            : I(0)
        {
        }

        void SetUp() override {
            I = 5;
        }

        int I;
    };

    struct TSimpleFixture : public NUnitTest::TBaseFixture {
        size_t Value = 24;
    };

    struct TOtherFixture : public NUnitTest::TBaseFixture {
        size_t TheAnswer = 42;
    };
}

TEST_F(TFixture, Test1) {
    ASSERT_EQ(I, 5);
}

TEST(ETest, Test1) {
    UNIT_CHECK_GENERATED_EXCEPTION(ythrow yexception(), yexception);
    UNIT_CHECK_GENERATED_NO_EXCEPTION(true, yexception);
}

Y_UNIT_TEST_SUITE(TestSingleTestFixture)
{
    Y_UNIT_TEST_F(Test3, TSimpleFixture) {
        UNIT_ASSERT_EQUAL(Value, 24);
    }
}

Y_UNIT_TEST_SUITE_F(TestSuiteFixture, TSimpleFixture)
{
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT(Value == 24);
        Value = 25;
    }

    Y_UNIT_TEST(Test2) {
        UNIT_ASSERT_EQUAL(Value, 24);
    }

    Y_UNIT_TEST_F(Test3, TOtherFixture) {
        UNIT_ASSERT_EQUAL(TheAnswer, 42);
    }
}
