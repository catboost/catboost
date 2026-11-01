#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/atomic_object.h>

#include <util/generic/string.h>

#include <thread>
#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TValue
{
    TValue() = default;

    explicit TValue(int payload)
        : Payload(payload)
    { }

    bool operator == (const TValue& other) const = default;

    int Payload = 0;
};

////////////////////////////////////////////////////////////////////////////////

TEST(TAtomicObjectTest, DefaultConstructed)
{
    TAtomicObject<TValue> object;
    EXPECT_EQ(object.Load(), TValue(0));
}

TEST(TAtomicObjectTest, ConstructedFromValue)
{
    TAtomicObject<TValue> object(TValue(42));
    EXPECT_EQ(object.Load(), TValue(42));
}

TEST(TAtomicObjectTest, Store)
{
    TAtomicObject<TString> object;
    object.Store("hello");
    EXPECT_EQ(object.Load(), "hello");

    TString value = "world";
    object.Store(value);
    EXPECT_EQ(object.Load(), "world");
    EXPECT_EQ(value, "world");
}

TEST(TAtomicObjectTest, StoreMoves)
{
    TAtomicObject<TString> object;

    TString value(1000, 'x');
    const auto* data = value.data();
    object.Store(std::move(value));

    const auto* storedData = object.Read([] (const TString& value) {
        return value.data();
    });
    EXPECT_EQ(storedData, data);
}

TEST(TAtomicObjectTest, Exchange)
{
    TAtomicObject<TValue> object(TValue(1));

    EXPECT_EQ(object.Exchange(TValue(2)), TValue(1));
    EXPECT_EQ(object.Load(), TValue(2));
    EXPECT_EQ(object.Exchange(TValue(3)), TValue(2));
    EXPECT_EQ(object.Load(), TValue(3));
}

TEST(TAtomicObjectTest, CompareExchangeSucceeds)
{
    TAtomicObject<TValue> object(TValue(1));

    auto expected = TValue(1);
    EXPECT_TRUE(object.CompareExchange(expected, TValue(2)));
    EXPECT_EQ(expected, TValue(1));
    EXPECT_EQ(object.Load(), TValue(2));
}

TEST(TAtomicObjectTest, CompareExchangeFailsAndUpdatesExpected)
{
    TAtomicObject<TValue> object(TValue(1));

    auto expected = TValue(7);
    EXPECT_FALSE(object.CompareExchange(expected, TValue(2)));
    EXPECT_EQ(expected, TValue(1));
    EXPECT_EQ(object.Load(), TValue(1));

    EXPECT_TRUE(object.CompareExchange(expected, TValue(2)));
    EXPECT_EQ(object.Load(), TValue(2));
}

TEST(TAtomicObjectTest, Transform)
{
    TAtomicObject<TValue> object(TValue(1));

    object.Transform([] (TValue& value) {
        value.Payload += 10;
    });
    EXPECT_EQ(object.Load(), TValue(11));

    auto result = object.Transform([] (TValue& value) {
        value.Payload *= 2;
        return value.Payload;
    });
    EXPECT_EQ(result, 22);
    EXPECT_EQ(object.Load(), TValue(22));
}

TEST(TAtomicObjectTest, Read)
{
    const TAtomicObject<TValue> object(TValue(3));

    auto payload = object.Read([] (const TValue& value) {
        return value.Payload;
    });
    EXPECT_EQ(payload, 3);
}

TEST(TAtomicObjectTest, ConcurrentTransform)
{
    constexpr int ThreadCount = 4;
    constexpr int IterationCount = 10000;

    TAtomicObject<TValue> object;

    std::vector<std::thread> threads;
    for (int index = 0; index < ThreadCount; ++index) {
        threads.emplace_back([&] {
            for (int iteration = 0; iteration < IterationCount; ++iteration) {
                object.Transform([] (TValue& value) {
                    ++value.Payload;
                });
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(object.Load(), TValue(ThreadCount * IterationCount));
}

TEST(TAtomicObjectTest, ConcurrentCompareExchange)
{
    constexpr int ThreadCount = 4;
    constexpr int IterationCount = 10000;

    TAtomicObject<TValue> object;

    std::vector<std::thread> threads;
    for (int index = 0; index < ThreadCount; ++index) {
        threads.emplace_back([&] {
            for (int iteration = 0; iteration < IterationCount; ++iteration) {
                auto expected = object.Load();
                while (!object.CompareExchange(expected, TValue(expected.Payload + 1))) {
                    // NB: #expected has been refreshed by the failed exchange.
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(object.Load(), TValue(ThreadCount * IterationCount));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
