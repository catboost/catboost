#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/count_down_latch.h>

#include <thread>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

void WaitForLatch(const TCountDownLatch& latch)
{
    latch.Wait();
    EXPECT_EQ(0, latch.GetCount());
}

TEST(TCountDownLatch, TwoThreads)
{
    TCountDownLatch latch(2);

    std::jthread t1(std::bind(&WaitForLatch, std::cref(latch)));
    std::jthread t2(std::bind(&WaitForLatch, std::cref(latch)));

    EXPECT_EQ(2, latch.GetCount());
    latch.CountDown();
    EXPECT_EQ(1, latch.GetCount());
    latch.CountDown();
    EXPECT_EQ(0, latch.GetCount());
}

TEST(TCountDownLatch, TwoThreadsPredecremented)
{
    TCountDownLatch latch(2);

    EXPECT_EQ(2, latch.GetCount());
    latch.CountDown();
    EXPECT_EQ(1, latch.GetCount());
    latch.CountDown();
    EXPECT_EQ(0, latch.GetCount());

    std::jthread t1(std::bind(&WaitForLatch, std::cref(latch)));
    std::jthread t2(std::bind(&WaitForLatch, std::cref(latch)));
}

TEST(TCountDownLatch, TwoThreadsTwoLatches)
{
    TCountDownLatch first(1);
    TCountDownLatch second(1);

    std::jthread t1([&] () {
        first.Wait();
        second.CountDown();
        EXPECT_EQ(0, first.GetCount());
        EXPECT_EQ(0, second.GetCount());
    });

    std::jthread t2([&] () {
        first.CountDown();
        second.Wait();
        EXPECT_EQ(0, first.GetCount());
        EXPECT_EQ(0, second.GetCount());
    });
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
