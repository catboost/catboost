#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/atomic.h>

#include <thread>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TAtomicTest, SingleWriterFetchAdd)
{
    std::atomic<int> value;

    // Test basic addition
    int oldValue = SingleWriterFetchAdd(value, 5);
    EXPECT_EQ(oldValue, 0);
    EXPECT_EQ(value.load(), 5);

    oldValue = SingleWriterFetchAdd(value, 10);
    EXPECT_EQ(oldValue, 5);
    EXPECT_EQ(value.load(), 15);

    // Test with negative delta
    oldValue = SingleWriterFetchAdd(value, -3);
    EXPECT_EQ(oldValue, 15);
    EXPECT_EQ(value.load(), 12);
}

TEST(TAtomicTest, SingleWriterFetchSub)
{
    std::atomic<int> value{100};

    // Test basic subtraction
    int oldValue = SingleWriterFetchSub(value, 5);
    EXPECT_EQ(oldValue, 100);
    EXPECT_EQ(value.load(), 95);

    oldValue = SingleWriterFetchSub(value, 10);
    EXPECT_EQ(oldValue, 95);
    EXPECT_EQ(value.load(), 85);

    // Test with negative delta (which is effectively addition)
    oldValue = SingleWriterFetchSub(value, -3);
    EXPECT_EQ(oldValue, 85);
    EXPECT_EQ(value.load(), 88);
}

TEST(TAtomicTest, SingleWriterFetchAddUnsigned)
{
    std::atomic<unsigned int> value;

    unsigned int oldValue = SingleWriterFetchAdd(value, 5u);
    EXPECT_EQ(oldValue, 0u);
    EXPECT_EQ(value.load(), 5u);

    oldValue = SingleWriterFetchAdd(value, 10u);
    EXPECT_EQ(oldValue, 5u);
    EXPECT_EQ(value.load(), 15u);
}

TEST(TAtomicTest, SingleWriterFetchSubUnsigned)
{
    std::atomic<unsigned int> value{100};

    unsigned int oldValue = SingleWriterFetchSub(value, 5u);
    EXPECT_EQ(oldValue, 100u);
    EXPECT_EQ(value.load(), 95u);

    oldValue = SingleWriterFetchSub(value, 10u);
    EXPECT_EQ(oldValue, 95u);
    EXPECT_EQ(value.load(), 85u);
}

TEST(TAtomicTest, SingleWriterFetchAddMultipleOperations)
{
    std::atomic<int> value;

    for (int i = 0; i < 100; ++i) {
        int oldValue = SingleWriterFetchAdd(value, i);
        EXPECT_EQ(oldValue, i * (i - 1) / 2);
    }

    EXPECT_EQ(value.load(), 4950); // Sum of 0..99
}

TEST(TAtomicTest, SingleWriterFetchSubMultipleOperations)
{
    std::atomic<int> value{4950};

    for (int i = 0; i < 100; ++i) {
        int oldValue = SingleWriterFetchSub(value, i);
        EXPECT_EQ(oldValue, 4950 - i * (i - 1) / 2);
    }

    EXPECT_EQ(value.load(), 0);
}

TEST(TAtomicTest, SingleWriterFetchAddWithZero)
{
    std::atomic<int> value{42};

    int oldValue = SingleWriterFetchAdd(value, 0);
    EXPECT_EQ(oldValue, 42);
    EXPECT_EQ(value.load(), 42);
}

TEST(TAtomicTest, SingleWriterFetchSubWithZero)
{
    std::atomic<int> value{42};

    int oldValue = SingleWriterFetchSub(value, 0);
    EXPECT_EQ(oldValue, 42);
    EXPECT_EQ(value.load(), 42);
}

TEST(TAtomicTest, SingleWriterFetchAddLargeDelta)
{
    std::atomic<int64_t> value;

    int64_t oldValue = SingleWriterFetchAdd(value, int64_t{1000000000});
    EXPECT_EQ(oldValue, 0);
    EXPECT_EQ(value.load(), int64_t{1000000000});

    oldValue = SingleWriterFetchAdd(value, int64_t{2000000000});
    EXPECT_EQ(oldValue, int64_t{1000000000});
    EXPECT_EQ(value.load(), int64_t{3000000000});
}

TEST(TAtomicTest, SingleWriterFetchSubLargeDelta)
{
    std::atomic<int64_t> value{int64_t{3000000000}};

    int64_t oldValue = SingleWriterFetchSub(value, int64_t{1000000000});
    EXPECT_EQ(oldValue, int64_t{3000000000});
    EXPECT_EQ(value.load(), int64_t{2000000000});

    oldValue = SingleWriterFetchSub(value, int64_t{2000000000});
    EXPECT_EQ(oldValue, int64_t{2000000000});
    EXPECT_EQ(value.load(), 0);
}

TEST(TAtomicTest, SingleWriterFetchAddConsistency)
{
    std::atomic<int> value;
    int sum = 0;

    for (int i = 1; i <= 1000; ++i) {
        int oldValue = SingleWriterFetchAdd(value, i);
        sum += i;
        EXPECT_EQ(oldValue, sum - i);
    }

    EXPECT_EQ(value.load(), sum);
}

TEST(TAtomicTest, SingleWriterFetchSubConsistency)
{
    std::atomic<int> value{500500}; // Sum of 1..1000
    int sum = 500500;

    for (int i = 1; i <= 1000; ++i) {
        int oldValue = SingleWriterFetchSub(value, i);
        EXPECT_EQ(oldValue, sum);
        sum -= i;
    }

    EXPECT_EQ(value.load(), 0);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
