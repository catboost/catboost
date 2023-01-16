#include <library/cpp/iterator/mapped.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/map.h>
#include <util/generic/vector.h>

using namespace testing;

TEST(TIterator, TMappedIteratorTest) {
    TVector<int> x = {1, 2, 3, 4, 5};
    auto it = MakeMappedIterator(x.begin(), [](int x) { return x + 7; });

    EXPECT_EQ(*it, 8);
    EXPECT_EQ(it[2], 10);
}

TEST(TIterator, TMappedRangeTest) {
    TVector<int> x = {1, 2, 3, 4, 5};

    EXPECT_THAT(
        MakeMappedRange(
            x,
            [](int x) { return x + 3; }
        ),
        ElementsAre(4, 5, 6, 7, 8)
    );
}

//TODO: replace with dedicated IterateKeys / IterateValues methods
TEST(TIterator, TMutableMappedRangeTest) {
    TMap<int, int> points = {{1, 2}, {3, 4}};

    EXPECT_THAT(
        MakeMappedRange(
            points.begin(), points.end(),
            [](TMap<int, int>::value_type& kv) -> int& { return kv.second; }
        ),
        ElementsAre(2, 4)
    );
}

TEST(TIterator, TOwningMappedMethodTest) {
    auto range = MakeMappedRange(
        TVector<std::pair<int, int>>{std::make_pair(1, 2), std::make_pair(3, 4)},
        [](auto& point) -> int& {
            return point.first;
        }
    );
    EXPECT_EQ(range[0], 1);
    range[0] += 1;
    EXPECT_EQ(range[0], 2);
    (*range.begin()) += 1;
    EXPECT_EQ(range[0], 3);
    for (int& y : range) {
        y += 7;
    }

    EXPECT_EQ(*range.begin(), 10);
    EXPECT_EQ(*(range.begin() + 1), 10);
}
