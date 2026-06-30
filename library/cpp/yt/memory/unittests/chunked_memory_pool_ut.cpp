#include <util/string/cast.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/chunked_memory_pool.h>

namespace NYT {
namespace {

using ::ToString;

////////////////////////////////////////////////////////////////////////////////

TEST(TChunkedMemoryPoolTest, Absorb)
{
    TChunkedMemoryPool first;
    TChunkedMemoryPool second;
    TChunkedMemoryPool third;

    std::vector<std::pair<TStringBuf, TString>> tests;
    size_t totalSize = 0;

    auto fillPool = [&] (TChunkedMemoryPool& pool, TString prefix, int count) {
        for (int i = 0; i < count; i++) {
            TString expected = prefix + ToString(count);
            char* buf = pool.AllocateUnaligned(expected.size());
            ::memcpy(buf, expected.c_str(), expected.size());
            TStringBuf ref(buf, buf + expected.size());
            totalSize += expected.size();
            tests.emplace_back(std::move(ref), std::move(expected));
        }
    };

    auto checkAll = [&] {
        ASSERT_GE(first.GetCapacity(), first.GetSize());
        ASSERT_GE(second.GetCapacity(), second.GetSize());
        ASSERT_GE(third.GetCapacity(), third.GetSize());
        ASSERT_EQ(totalSize, first.GetSize() + second.GetSize() + third.GetSize());

        for (const auto& [ref, expected] : tests) {
            ASSERT_EQ(ref, expected);
        }
    };

    auto warmup = [&] (TChunkedMemoryPool& pool, int blockSize, int count) {
        ASSERT_EQ(pool.GetSize(), 0U);
        for (int i = 0; i < count; i++) {
            pool.AllocateUnaligned(blockSize);
        }
        pool.Clear();
        ASSERT_EQ(pool.GetSize(), 0U);
    };

    warmup(first, 20, 20'000);
    warmup(second, 20, 20'000);
    fillPool(first, "firstPool", 10'000);
    fillPool(second, "secondPool", 10'000);
    fillPool(third, "thirdPool", 10'000);
    checkAll();
    second.Absorb(std::move(third));
    checkAll();
    first.Absorb(std::move(second));
    checkAll();
    fillPool(first, "poolFirst_2", 10'000);
    fillPool(second, "poolSecond_2", 10'000);
    fillPool(third, "poolThird_2", 10'000);
    checkAll();
    third.Absorb(std::move(second));
    checkAll();
    fillPool(second, "someStr2", 10'000);
    fillPool(third, "someStr3", 10'000);
    checkAll();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
