#pragma once

// WARNING: this is a legacy header that tries to mimic the gtest interface while using unittest
// under the hood. Avoid using this interface -- use the genuine gtest instead (the GTEST macro).
// If you're already using GTEST macro and you've found yourself here, you probably meant
// to include `library/cpp/testing/gtest/gtest.h`.

#include "registar.h"

#include <util/generic/ymath.h>
#include <util/generic/ylimits.h>

namespace NUnitTest {
    namespace NPrivate {
        struct IGTestFactory: public ITestBaseFactory {
            ~IGTestFactory() override;

            virtual void AddTest(const char* name, void (*body)(TTestContext&), bool forceFork) = 0;
        };

        IGTestFactory* ByName(const char* name);
    }
}

namespace NTesting {
    struct TTest {
        virtual void SetUp() {
        }

        virtual void TearDown() {
        }

        inline TTest* _This() noexcept {
            return this;
        }

        virtual ~TTest() = default;
    };
}

namespace testing {
    struct Test: public ::NTesting::TTest {
    };
}

#define TEST_IMPL(N, NN, FF)                                                         \
    void Test##N##NN(NUnitTest::TTestContext&);                                      \
    namespace NTestSuite##N##NN {                                                    \
        struct TReg {                                                                \
            inline TReg() {                                                          \
                ::NUnitTest::NPrivate::ByName(#N)->AddTest(#NN, &(Test##N##NN), FF); \
            }                                                                        \
        };                                                                           \
        static TReg reg;                                                             \
    }                                                                                \
    void Test##N##NN(NUnitTest::TTestContext&)

#define TEST_F_IMPL(N, NN, FF)                \
    namespace NTestSuite##N##NN {             \
        struct TTestSuite: public N {         \
            inline TTestSuite() {             \
                this->_This()->SetUp();       \
            }                                 \
            inline ~TTestSuite() {            \
                this->_This()->TearDown();    \
            }                                 \
            void NN();                        \
        };                                    \
    }                                         \
    TEST_IMPL(N, NN, FF) {                    \
        NTestSuite##N##NN::TTestSuite().NN(); \
    }                                         \
    void NTestSuite##N##NN::TTestSuite::NN()

#define TEST(A, B) TEST_IMPL(A, B, false)
#define TEST_FORKED(A, B) TEST_IMPL(A, B, true)

#define TEST_F(A, B) TEST_F_IMPL(A, B, false)
#define TEST_F_FORKED(A, B) TEST_F_IMPL(A, B, true)

#define EXPECT_EQ(A, B) UNIT_ASSERT_VALUES_EQUAL(A, B)
#define EXPECT_NE(A, B) UNIT_ASSERT_UNEQUAL(A, B)
#define EXPECT_LE(A, B) UNIT_ASSERT((A) <= (B))
#define EXPECT_LT(A, B) UNIT_ASSERT((A) < (B))
#define EXPECT_GE(A, B) UNIT_ASSERT((A) >= (B))
#define EXPECT_GT(A, B) UNIT_ASSERT((A) > (B))
#define EXPECT_NO_THROW(A) UNIT_ASSERT_NO_EXCEPTION(A)
#define EXPECT_THROW(A, B) UNIT_ASSERT_EXCEPTION(A, B)
#define EXPECT_NEAR(A, B, D) UNIT_ASSERT_DOUBLES_EQUAL(A, B, D)
#define EXPECT_STREQ(A, B) UNIT_ASSERT_VALUES_EQUAL(A, B)

#define EXPECT_DOUBLE_EQ_TOLERANCE(A, B, tolerance) UNIT_ASSERT_C(fabs((A) - (B)) < tolerance * std::numeric_limits<decltype(A)>::epsilon(), TString("\n") + ToString(A) + " <> " + ToString(B))
#define EXPECT_DOUBLE_EQ(A, B) EXPECT_DOUBLE_EQ_TOLERANCE(A, B, 4.0)

//conflicts with util/system/defaults.h
#undef EXPECT_TRUE
#define EXPECT_TRUE(X) UNIT_ASSERT(X)
#undef EXPECT_FALSE
#define EXPECT_FALSE(X) UNIT_ASSERT(!(X))

#define ASSERT_EQ(A, B) EXPECT_EQ(A, B)
#define ASSERT_NE(A, B) EXPECT_NE(A, B)
#define ASSERT_GT(A, B) EXPECT_GT(A, B)
#define ASSERT_LT(A, B) EXPECT_LT(A, B)
#define ASSERT_FALSE(X) EXPECT_FALSE(X)
#define ASSERT_TRUE(X) EXPECT_TRUE(X)
#define ASSERT_THROW(A, B) EXPECT_THROW(A, B)
#define ASSERT_NO_THROW(A) EXPECT_NO_THROW(A)
#define ASSERT_DOUBLE_EQ(A, B) EXPECT_DOUBLE_EQ(A, B)
#define ASSERT_STREQ(A, B) EXPECT_STREQ(A, B)
